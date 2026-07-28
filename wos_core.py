"""
Core Monte Carlo PDE solvers for fluid simulation comparison.
Implements WoS (2022) and WoB/velocity-based (2024) approaches.

References:
  [2022] Rioux-Lavoie et al. "A Monte Carlo Method for Fluid Simulation"
  [2024] Sugimoto et al. "Velocity-Based Monte Carlo Fluids"
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass
import time


@dataclass
class Domain2D:
    """2D domain with boundary distance and condition queries"""
    boundary_dist: Callable  # (N,2) -> (N,) distance to nearest boundary (positive inside)
    inside: Callable         # (N,2) -> (N,) boolean mask
    boundary_normal: Callable  # (N,2) -> (N,2) outward normal at nearest boundary point
    dirichlet_bc: Callable   # (N,2) -> (N,) or (N,2) Dirichlet value at boundary point
    bounding_box: tuple      # (min_x, max_x, min_y, max_y)


def rect_obstacle_domain(rect_size: float = 2.0,
                          obs_center=(0.0, 0.0),
                          obs_radius=0.2) -> Domain2D:
    """Rectangle domain with a circular obstacle inside.
    Outer boundary: Dirichlet u=0.
    Obstacle surface: Dirichlet u=0 (no-slip).
    """
    xmin, xmax = -rect_size/2, rect_size/2
    ymin, ymax = -rect_size/2, rect_size/2
    oc = np.array(obs_center)
    orad = obs_radius

    def boundary_dist(pts):
        d_rect = _signed_dist_to_rect(pts, xmin, xmax, ymin, ymax)
        d_obs = np.linalg.norm(pts - oc, axis=-1) - orad
        # Absolute distances
        d_rect_abs = np.abs(d_rect)
        d_obs_abs = np.abs(d_obs)
        inside_rect = _inside_rect(pts, xmin, xmax, ymin, ymax)
        outside_obs = d_obs >= 0.0
        in_domain = inside_rect & outside_obs
        dist_abs = np.minimum(d_rect_abs, d_obs_abs)
        return np.where(in_domain, dist_abs, -dist_abs)

    def inside(pts):
        in_rect = _inside_rect(pts, xmin, xmax, ymin, ymax)
        out_obs = np.linalg.norm(pts - oc, axis=-1) >= orad
        return in_rect & out_obs

    def bc_dirichlet(pts):
        return np.zeros(pts.shape[:-1])

    def boundary_normal_fn(pts):
        n = np.zeros_like(pts)
        near_left = np.abs(pts[:, 0] - xmin) < 0.01
        near_right = np.abs(pts[:, 0] - xmax) < 0.01
        near_bottom = np.abs(pts[:, 1] - ymin) < 0.01
        near_top = np.abs(pts[:, 1] - ymax) < 0.01
        n[near_left, 0] = -1.0
        n[near_right, 0] = 1.0
        n[near_bottom, 1] = -1.0
        n[near_top, 1] = 1.0
        d_obs = np.linalg.norm(pts - oc, axis=-1)
        near_obs = np.abs(d_obs - orad) < 0.02
        if np.any(near_obs):
            denom = np.maximum(d_obs[near_obs], 1e-10)
            n[near_obs] = (pts[near_obs] - oc) / denom[:, None]
        norms = np.linalg.norm(n, axis=-1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return n / norms

    return Domain2D(boundary_dist, inside, boundary_normal_fn, bc_dirichlet,
                    (xmin, xmax, ymin, ymax))


def _signed_dist_to_rect(pts, xmin, xmax, ymin, ymax):
    """Signed distance to rectangle: positive outside, negative inside."""
    dx = np.maximum(xmin - pts[:, 0], 0.0)
    dx = np.maximum(dx, pts[:, 0] - xmax)
    dy = np.maximum(ymin - pts[:, 1], 0.0)
    dy = np.maximum(dy, pts[:, 1] - ymax)
    outside_dist = np.sqrt(dx**2 + dy**2)

    inside_dist = np.minimum.reduce([
        pts[:, 0] - xmin, xmax - pts[:, 0],
        pts[:, 1] - ymin, ymax - pts[:, 1]
    ])
    return np.where(outside_dist > 0, outside_dist, -inside_dist)


def _inside_rect(pts, xmin, xmax, ymin, ymax):
    return (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & \
           (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)


# ─── Walk-on-Spheres Solver ───────────────────────────────────────────────

def wos_poisson_2d(pts: np.ndarray,
                    source: Callable[[np.ndarray], np.ndarray],
                    domain: Domain2D,
                    n_walks: int = 512,
                    eps: float = 1e-3,
                    max_steps: int = 500,
                    neumann_bc: bool = True) -> np.ndarray:
    """
    Walk-on-Spheres for Δp = f in Ω.

    Neumann BC (∂p/∂n = 0): reflect at boundaries, accumulate source only.
      p(x) = -E[∫₀^τ f(B_s)/2 ds] up to additive constant.
      The constant is fixed by setting mean(p) = 0.

    Dirichlet BC: p = g on ∂Ω, use standard termination.
      p(x) = E[g(x_τ)] - ½E[∫₀^τ f(B_s) ds]
    """
    rng = np.random.default_rng()
    flat = pts.reshape(-1, 2)
    n = flat.shape[0]
    result = np.zeros(n, dtype=np.float64)

    for walk_i in range(n_walks):
        pos = flat.copy()
        accum = np.zeros(n, dtype=np.float64)
        alive = np.ones(n, dtype=bool)

        for step in range(max_steps):
            if not np.any(alive):
                break

            d = domain.boundary_dist(pos)
            near = d < eps

            if neumann_bc:
                # Neumann: reflect at boundary, continue accumulating
                # In the reflection case, we keep walking
                reflect = near & alive
                if np.any(reflect):
                    r_idx = np.where(reflect)[0]
                    nrm = domain.boundary_normal(pos[reflect])
                    step_back = np.zeros((len(r_idx), 2))
                    for k, idx in enumerate(r_idx):
                        nk = nrm[k]
                        nk_norm = np.linalg.norm(nk)
                        if nk_norm > 0:
                            nk = nk / nk_norm
                            # Reflect the walk: push back into domain
                            step_back[k] = nk * (d[idx] + eps)
                    pos[reflect] += step_back
                    alive[reflect] = True  # stay alive
            else:
                # Dirichlet: terminate at boundary
                done = near & alive
                if np.any(done):
                    bc = domain.dirichlet_bc(pos[done])
                    if isinstance(bc, np.ndarray) and bc.ndim > 0:
                        result[done] += bc - accum[done]
                    alive[done] = False

            if not np.any(alive):
                break

            na = np.sum(alive)
            theta = rng.uniform(0, 2 * np.pi, size=na)
            step_vec = np.column_stack([np.cos(theta), np.sin(theta)])
            step_vec *= d[alive, None]

            f_vals = source(pos[alive])
            accum[alive] += f_vals * (d[alive] ** 2) / 4.0

            pos[alive] += step_vec

        if np.any(alive):
            if neumann_bc:
                result[alive] -= accum[alive]
            else:
                result[alive] += domain.dirichlet_bc(pos[alive]) - accum[alive]

    result /= n_walks

    if neumann_bc:
        # Fix additive constant: mean(p) = 0
        result -= np.mean(result)

    return result.reshape(pts.shape[:-1])


def wos_diffuse_2d(pts: np.ndarray,
                    field_at: Callable[[np.ndarray], np.ndarray],
                    domain: Domain2D,
                    dt: float,
                    nu: float,
                    n_samples: int = 256) -> np.ndarray:
    """
    Monte Carlo diffusion via Feynman-Kac formula.

    Solves ∂u/∂t = ν∇²u.

    u(x, t+dt) = E[ u(x + √(2νdt) Z, t) ]  where Z ~ N(0,I)

    If the random step exits the domain, apply reflection or Dirichlet BC.
    """
    rng = np.random.default_rng()
    flat = pts.reshape(-1, 2)
    n = flat.shape[0]
    sigma = np.sqrt(2.0 * nu * dt)
    eps = 1e-4

    sample = field_at(flat[:1])
    scalar = not isinstance(sample, np.ndarray) or sample.ndim == 0
    dim = 1 if scalar else sample.shape[-1]

    result = np.zeros((n, dim) if dim > 1 else n, dtype=np.float64)

    for i in range(n):
        vals = np.zeros((n_samples, dim) if dim > 1 else n_samples, dtype=np.float64)

        for s in range(n_samples):
            pos = flat[i].copy()

            for hop in range(50):
                step = rng.normal(0, sigma, size=2)
                new_pos = pos + step

                nd = domain.boundary_dist(new_pos[None])[0]

                if nd >= eps:
                    # Inside domain: record field value at new position
                    vals[s] = field_at(new_pos[None])[0]
                    break

                elif nd >= 0:
                    # On boundary: apply Dirichlet BC
                    bc = domain.dirichlet_bc(new_pos[None])
                    vals[s] = bc[0] if isinstance(bc, np.ndarray) else bc
                    break

                else:
                    # Outside domain: reflect
                    nrm = domain.boundary_normal(pos[None])[0]
                    nrm_norm = np.linalg.norm(nrm)
                    if nrm_norm > 0:
                        step -= 2 * np.dot(step, nrm / nrm_norm) * (nrm / nrm_norm)
                    new_pos = pos + step
                    nd = domain.boundary_dist(new_pos[None])[0]
                    if nd >= 0:
                        if nd >= eps:
                            vals[s] = field_at(new_pos[None])[0]
                        else:
                            bc = domain.dirichlet_bc(new_pos[None])
                            vals[s] = bc[0] if isinstance(bc, np.ndarray) else bc
                        break
                    pos = new_pos

        result[i] = np.mean(vals, axis=0)

    return result.reshape(pts.shape[:-1] + ((dim,) if dim > 1 else ()))


# ─── Velocity-based Monte Carlo Projection (2024 paper) ───────────────────

def wob_project_velocity_2d(pts: np.ndarray,
                             velocity_grid: np.ndarray,
                             domain: Domain2D,
                             n_walks: int = 256) -> np.ndarray:
    """
    Walk-on-Boundary velocity projection for incompressibility (2024 approach).

    Directly estimates the divergence-free projected velocity using Monte Carlo.
    
    P(u)(x) = u(x) - E[ boundary correction from WoS walks ]
    
    For each WoS walk from x to boundary, the projected velocity at x is
    estimated by averaging the velocity sampled along the walk. Walks that
    hit the boundary contribute the boundary value (no-slip = 0).
    """
    rng = np.random.default_rng()
    flat = pts.reshape(-1, 2)
    n = flat.shape[0]
    result = np.zeros((n, 2), dtype=np.float64)

    for i in range(n):
        x0 = flat[i]

        for s in range(n_walks):
            pos = x0.copy()
            vel_sum = np.zeros(2)
            count = 0

            for hop in range(50):
                d = domain.boundary_dist(pos[None])[0]

                if d < 1e-3:
                    # Hit boundary: no-slip (u=0), contribute zero
                    break

                # WoS step
                theta = rng.uniform(0, 2 * np.pi)
                new_pos = pos + d * np.array([np.cos(theta), np.sin(theta)])

                nd = domain.boundary_dist(new_pos[None])[0]
                if nd < 1e-3:
                    # New position is on or very near boundary
                    break

                # Sample velocity at the new position
                vi = _interp2d(new_pos, velocity_grid)
                vel_sum += vi
                count += 1
                pos = new_pos

            if count > 0:
                result[i] += vel_sum / count

        result[i] /= n_walks

    return result.reshape(pts.shape[:-1] + (2,))


# ─── Grid utilities ───────────────────────────────────────────────────────

def _interp2d(pos: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on [-1,1]² grid."""
    shape = grid.shape[:2]
    ix = (pos[0] + 1.0) * (shape[0] - 1) / 2.0
    iy = (pos[1] + 1.0) * (shape[1] - 1) / 2.0
    i, j = int(np.floor(ix)), int(np.floor(iy))
    fi, fj = ix - i, iy - j
    i = np.clip(i, 0, shape[0] - 2)
    j = np.clip(j, 0, shape[1] - 2)
    if grid.ndim == 2:
        return ((1-fi)*(1-fj)*grid[i, j] + fi*(1-fj)*grid[i+1, j] +
                (1-fi)*fj*grid[i, j+1] + fi*fj*grid[i+1, j+1])
    else:
        v = ((1-fi)*(1-fj)*grid[i, j] + fi*(1-fj)*grid[i+1, j] +
             (1-fi)*fj*grid[i, j+1] + fi*fj*grid[i+1, j+1])
        return v


def divergence_2d(vx: np.ndarray, vy: np.ndarray, h: float) -> np.ndarray:
    """Central difference divergence."""
    div = np.zeros_like(vx)
    div[1:-1, :] += (vx[2:, :] - vx[:-2, :]) / (2 * h)
    div[:, 1:-1] += (vy[:, 2:] - vy[:, :-2]) / (2 * h)
    return div


def gradient_2d(f: np.ndarray, h: float):
    """Central difference gradient."""
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * h)
    gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    return gx, gy


def curl_2d(vx: np.ndarray, vy: np.ndarray, h: float) -> np.ndarray:
    """2D vorticity ω = ∂v_y/∂x - ∂v_x/∂y"""
    curl = np.zeros_like(vx)
    curl[1:-1, 1:-1] = (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * h) - \
                        (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * h)
    return curl


def gaussian_smooth(field: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Apply Gaussian smoothing to reduce Monte Carlo noise."""
    from scipy.ndimage import gaussian_filter
    if field.ndim == 3:
        result = np.zeros_like(field)
        for c in range(field.shape[-1]):
            result[..., c] = gaussian_filter(field[..., c], sigma=sigma, mode='reflect')
        return result
    return gaussian_filter(field, sigma=sigma, mode='reflect')


def semi_lagrangian_advect(field: np.ndarray,
                            velocity: np.ndarray,
                            dt: float) -> np.ndarray:
    """Semi-Lagrangian advection on [-1,1]² grid."""
    shape = field.shape[:2]
    result = np.zeros_like(field)

    for i in range(shape[0]):
        for j in range(shape[1]):
            x = -1.0 + 2.0 * i / (shape[0] - 1)
            y = -1.0 + 2.0 * j / (shape[1] - 1)

            v = _interp2d(np.array([x, y]), velocity)
            xb = x - v[0] * dt
            yb = y - v[1] * dt

            if field.ndim == 2:
                result[i, j] = _interp2d(np.array([xb, yb]), field)
            else:
                for c in range(field.shape[-1]):
                    result[i, j, c] = _interp2d(np.array([xb, yb]), field[..., c])

    return result


def apply_no_slip(velocity: np.ndarray, domain: Domain2D, grid_res: int):
    """Enforce no-slip boundary condition on obstacle."""
    for i in range(grid_res):
        for j in range(grid_res):
            x = -1.0 + 2.0 * i / (grid_res - 1)
            y = -1.0 + 2.0 * j / (grid_res - 1)
            if not domain.inside(np.array([[x, y]]))[0]:
                velocity[i, j] = 0.0
    return velocity


def taylor_green_vortex(grid_res: int, t: float = 0.0, nu: float = 0.1) -> np.ndarray:
    """Taylor-Green decaying vortices as a test case."""
    x = np.linspace(-1, 1, grid_res)
    y = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    decay = np.exp(-2 * np.pi**2 * nu * t)
    vx = -np.cos(np.pi * X) * np.sin(np.pi * Y) * decay
    vy = np.sin(np.pi * X) * np.cos(np.pi * Y) * decay
    return np.stack([vx, vy], axis=-1)
