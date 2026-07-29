"""
Numba-accelerated WoS solvers for 2D fluid Monte Carlo.
Provides 50-100x speedup over pure Python loops.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _dist_to_rect_numba(x, y, xmin, xmax, ymin, ymax):
    """Signed distance to rectangle (positive outside, negative inside)."""
    # Outside distance
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    out_dist = np.sqrt(dx*dx + dy*dy)
    # Inside distance (distance to nearest wall)
    in_dist = min(x - xmin, xmax - x, y - ymin, ymax - y)
    if out_dist > 0:
        return out_dist
    return -in_dist


@njit(cache=True)
def _dist_to_obs_numba(x, y, ox, oy, orad):
    """Signed distance to circular obstacle."""
    return np.sqrt((x - ox)**2 + (y - oy)**2) - orad


@njit(cache=True)
def boundary_dist_rect_only(x, y):
    """UNSIGNED distance to [-1,1]² boundary (always >= 0 inside domain)."""
    inside = -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0
    if inside:
        return min(x + 1.0, 1.0 - x, y + 1.0, 1.0 - y)
    else:
        dx = max(-1.0 - x, 0.0, x - 1.0)
        dy = max(-1.0 - y, 0.0, y - 1.0)
        return np.sqrt(dx*dx + dy*dy)


@njit(cache=True)
def _inside_rect(x, y):
    return -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0


@njit(cache=True)
def boundary_dist_rect_obs(x, y, ox, oy, orad):
    """UNSIGNED distance for [-1,1]² with circular obstacle.
    Positive inside domain, negative outside."""
    in_rect = _inside_rect(x, y)
    d_obs = np.sqrt((x - ox)**2 + (y - oy)**2) - orad
    out_obs = d_obs >= 0.0

    if in_rect and out_obs:
        # Inside domain: distance = min(dist_to_rect, dist_to_obs)
        d_rect = min(x + 1.0, 1.0 - x, y + 1.0, 1.0 - y)
        return min(d_rect, d_obs)
    else:
        # Outside domain: negative distance
        if not in_rect:
            dx = max(-1.0 - x, 0.0, x - 1.0)
            dy = max(-1.0 - y, 0.0, y - 1.0)
            d_rect_out = np.sqrt(dx*dx + dy*dy)
        else:
            d_rect_out = 1e10
        d_out = d_rect_out
        if not out_obs:
            d_out = min(d_out, -d_obs)
        return -d_out


@njit(cache=True)
def boundary_normal_rect_obs(x, y, ox, oy, orad):
    """Outward normal at nearest boundary point."""
    # Check which boundary is nearer
    dr = _dist_to_rect_numba(x, y, -1.0, 1.0, -1.0, 1.0)
    do = _dist_to_obs_numba(x, y, ox, oy, orad)
    in_rect = -1 <= x <= 1 and -1 <= y <= 1
    out_obs = do >= 0

    near_rect = abs(dr) < abs(do) if (in_rect and out_obs) else True

    if near_rect:
        # Normal to rect
        nx, ny = 0.0, 0.0
        if abs(x - (-1.0)) < 0.02: nx = -1.0
        elif abs(x - 1.0) < 0.02: nx = 1.0
        if abs(y - (-1.0)) < 0.02: ny = -1.0
        elif abs(y - 1.0) < 0.02: ny = 1.0
    else:
        # Normal to obstacle (radially outward)
        dx, dy = x - ox, y - oy
        r = np.sqrt(dx*dx + dy*dy)
        if r > 0:
            nx, ny = dx / r, dy / r
        else:
            nx, ny = 1.0, 0.0

    norm = np.sqrt(nx*nx + ny*ny)
    if norm > 0:
        return nx / norm, ny / norm
    return 0.0, 0.0


@njit(cache=True)
def bilinear_interp(x, y, grid, res):
    """Bilinear interpolation on [-1,1]² grid."""
    ix = (x + 1.0) * (res - 1) / 2.0
    iy = (y + 1.0) * (res - 1) / 2.0
    i = int(np.floor(ix))
    j = int(np.floor(iy))
    fi = ix - i
    fj = iy - j
    i = max(0, min(i, res - 2))
    j = max(0, min(j, res - 2))
    return ((1-fi)*(1-fj)*grid[i, j] + fi*(1-fj)*grid[i+1, j] +
            (1-fi)*fj*grid[i, j+1] + fi*fj*grid[i+1, j+1])


@njit(cache=True)
def wos_poisson_point_numba(x, y, source_grid, res,
                              n_walks, eps, max_steps,
                              has_obs, ox, oy, orad,
                              use_neumann):
    """WoS Poisson solve at a single point."""
    total = 0.0

    for w in range(n_walks):
        px, py = x, y
        accum = 0.0
        alive = True

        for s in range(max_steps):
            if not alive:
                break

            if has_obs:
                d = boundary_dist_rect_obs(px, py, ox, oy, orad)
            else:
                d = boundary_dist_rect_only(px, py)

            if d < eps:
                if use_neumann:
                    nx_b, ny_b = boundary_normal_rect_obs(px, py, ox, oy, orad) if has_obs else (0.0, 0.0)
                    if abs(nx_b) > 0 or abs(ny_b) > 0:
                        px += -nx_b * (d + eps) * 2
                        py += -ny_b * (d + eps) * 2
                    else:
                        total += -accum
                        alive = False
                else:
                    total += -accum
                    alive = False
                continue

            # Source contribution BEFORE stepping (at current position)
            f_val = bilinear_interp(px, py, source_grid, res)
            accum += f_val * d * d / 4.0

            # WoS step
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            px += d * np.cos(theta)
            py += d * np.sin(theta)

        if alive:
            total += -accum if use_neumann else (0.0 - accum)

    return total / n_walks


@njit(parallel=True, cache=True)
def wos_poisson_grid_numba(X, Y, source_grid, res,
                            n_walks=256, eps=0.02, max_steps=500,
                            has_obs=False, ox=0.0, oy=0.0, orad=0.15,
                            use_neumann=False):
    """WoS Poisson solve for all grid points (parallel)."""
    n_pts = X.shape[0] * X.shape[1]
    result = np.zeros(X.shape, dtype=np.float64)

    for idx in prange(n_pts):
        i = idx // X.shape[1]
        j = idx % X.shape[1]
        x, y = X[i, j], Y[i, j]
        val = wos_poisson_point_numba(x, y, source_grid, res,
                                       n_walks, eps, max_steps,
                                       has_obs, ox, oy, orad,
                                       use_neumann)
        result[i, j] = val

    if use_neumann:
        result -= np.mean(result)

    return result


@njit(cache=True)
def wos_diffuse_point_numba(x, y, field_grid_u, field_grid_v, res,
                              dt, nu, n_samples):
    """WoS diffusion at a single point."""
    sigma = np.sqrt(2.0 * nu * dt)
    val_u, val_v = 0.0, 0.0
    eps_d = 1e-4

    for s in range(n_samples):
        px, py = x, y
        found = False

        for hop in range(50):
            step_x = np.random.normal(0.0, sigma)
            step_y = np.random.normal(0.0, sigma)
            nx, ny = px + step_x, py + step_y

            # Check if inside domain (simple rect check)
            if -1.0 <= nx <= 1.0 and -1.0 <= ny <= 1.0:
                val_u += bilinear_interp(nx, ny, field_grid_u, res)
                val_v += bilinear_interp(nx, ny, field_grid_v, res)
                found = True
                break
            else:
                # Bounce off boundary (simple reflection)
                if nx < -1.0: step_x = -step_x
                if nx > 1.0: step_x = -step_x
                if ny < -1.0: step_y = -step_y
                if ny > 1.0: step_y = -step_y
                px += step_x
                py += step_y

    if n_samples > 0:
        return val_u / n_samples, val_v / n_samples
    return 0.0, 0.0


@njit(parallel=True, cache=True)
def wos_diffuse_grid_numba(X, Y, field_u, field_v, res,
                            dt, nu, n_samples=256):
    """WoS diffusion for all grid points (parallel)."""
    n_pts = X.shape[0] * X.shape[1]
    result_u = np.zeros(X.shape, dtype=np.float64)
    result_v = np.zeros(X.shape, dtype=np.float64)

    for idx in prange(n_pts):
        i = idx // X.shape[1]
        j = idx % X.shape[1]
        x, y = X[i, j], Y[i, j]
        vu, vv = wos_diffuse_point_numba(x, y, field_u, field_v, res,
                                          dt, nu, n_samples)
        result_u[i, j] = vu
        result_v[i, j] = vv

    return result_u, result_v





# ═══════════════════════════════════════════════════════════════
# Walk-on-Boundary (WoB) Poisson solver & Projection
# Based on: Sugimoto et al. 2023 "Walk-on-Boundary Method"
#           Sugimoto et al. 2024 "Velocity-Based MC Fluids"
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _ray_intersect_rect_numba(x, y, dx, dy,
                               xmin=-1.0, xmax=1.0,
                               ymin=-1.0, ymax=1.0):
    """
    Ray-rectangle intersection in 2D.
    Returns (t, wall_id) where:
      t = distance from (x,y) to boundary in direction (dx,dy)
      wall_id: 0=right, 1=left, 2=top, 3=bottom
    If ray doesn't hit (shouldn't happen for interior point), t=inf, wall=-1.
    """
    t = 1e10
    wall = -1
    eps_d = 1e-12

    if dx > eps_d:
        tr = (xmax - x) / dx
        if tr > 0:
            yr = y + tr * dy
            if ymin - 1e-10 <= yr <= ymax + 1e-10 and tr < t:
                t, wall = tr, 0
    elif dx < -eps_d:
        tl = (xmin - x) / dx
        if tl > 0:
            yl = y + tl * dy
            if ymin - 1e-10 <= yl <= ymax + 1e-10 and tl < t:
                t, wall = tl, 1

    if dy > eps_d:
        tt = (ymax - y) / dy
        if tt > 0:
            xt = x + tt * dx
            if xmin - 1e-10 <= xt <= xmax + 1e-10 and tt < t:
                t, wall = tt, 2
    elif dy < -eps_d:
        tb = (ymin - y) / dy
        if tb > 0:
            xb = x + tb * dx
            if xmin - 1e-10 <= xb <= xmax + 1e-10 and tb < t:
                t, wall = tb, 3

    return t, wall


@njit(cache=True)
def _wob_particular_point(x, y, f_grid, res, n_samples):
    """
    Estimate p_p(x) = ∫_Ω G(x,y') f(y') dV(y') via Monte Carlo.
    G(x,y) = +(1/2π) * ln|x-y|  (2D free-space Green's function for ∇²G=δ)
    Samples uniformly in [-1,1]².
    """
    area = 4.0
    sum_val = 0.0
    inv_2pi = 0.15915494309189535

    for s in range(n_samples):
        sx = np.random.uniform(-1.0, 1.0)
        sy = np.random.uniform(-1.0, 1.0)
        f_val = bilinear_interp(sx, sy, f_grid, res)
        rx = x - sx
        ry = y - sy
        r2 = rx * rx + ry * ry
        if r2 < 1e-20:
            r2 = 1e-20
        sum_val += inv_2pi * np.log(np.sqrt(r2)) * f_val

    return sum_val * area / n_samples


@njit(cache=True)
def wob_poisson_point_numba(x, y, f_grid, res,
                             n_particular=64, n_rays=32):
    """
    Walk-on-Boundary Poisson solve at a single point.
    Solves ∇²p = f with p=0 on [-1,1]² rectangular boundary.

    Uses: p(x) = p_p(x) - E[p_p(y_boundary)]
    where:
      p_p(x) = ∫_Ω G(x,y') f(y') dV(y')  (free-space particular solution)
      y_boundary = first boundary hit point along a random ray from x
      The expectation E[p_p(y_boundary)] corrects p_p to satisfy p=0 on ∂Ω.

    This works because for a convex domain, the first-exit distribution
    of a random ray equals the harmonic measure (Poisson kernel).
    """
    # Step 1: Particular solution p_p(x) via MC volume integration
    p_p_x = _wob_particular_point(x, y, f_grid, res, n_particular)

    # Step 2: WoB boundary correction
    # p_h(x) = -E[p_p(y_boundary)]
    # where y_boundary is the first ray exit to boundary
    bc_sum = 0.0

    for ray in range(n_rays):
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        dx = np.cos(theta)
        dy = np.sin(theta)

        t, wall = _ray_intersect_rect_numba(x, y, dx, dy)

        if t > 1e9 or wall < 0:
            continue

        bx = x + t * dx
        by = y + t * dy

        p_p_b = _wob_particular_point(bx, by, f_grid, res,
                                       n_particular)
        bc_sum += p_p_b  # accumulates p_p at boundary

    bc_avg = bc_sum / n_rays

    # p(x) = p_p(x) + p_h(x) = p_p(x) - E[p_p(y_boundary)]
    return p_p_x - bc_avg


@njit(parallel=True, cache=True)
def wob_poisson_grid_numba(X, Y, f_grid, res,
                            n_particular=64, n_rays=32):
    """WoB Poisson solve for all grid points (parallel)."""
    n_pts = X.shape[0] * X.shape[1]
    result = np.zeros(X.shape, dtype=np.float64)

    for idx in prange(n_pts):
        i = idx // X.shape[1]
        j = idx % X.shape[1]
        x, y = X[i, j], Y[i, j]
        val = wob_poisson_point_numba(x, y, f_grid, res,
                                       n_particular, n_rays)
        result[i, j] = val

    return result


@njit(cache=True)
def wob_project_grid_numba(X, Y, u_grid, v_grid, res, h,
                            n_particular=64, n_rays=32):
    """
    Full WoB velocity projection on a grid.
    Returns (pressure, divergence_source) - smoothing and gradient
    done externally to avoid Numba/scipy conflict.
    """
    div = divergence_2d_numba(u_grid, v_grid, h)

    pressure = wob_poisson_grid_numba(X, Y, div, res,
                                       n_particular, n_rays)
    return pressure, div


@njit(cache=True)
def divergence_2d_numba(vx, vy, h):
    """Central difference divergence."""
    res = vx.shape[0]
    div = np.zeros((res, res))
    for i in range(1, res-1):
        for j in range(res):
            div[i, j] += (vx[i+1, j] - vx[i-1, j]) / (2 * h)
    for i in range(res):
        for j in range(1, res-1):
            div[i, j] += (vy[i, j+1] - vy[i, j-1]) / (2 * h)
    return div


@njit(cache=True)
def curl_2d_numba(vx, vy, h):
    """2D vorticity."""
    res = vx.shape[0]
    curl = np.zeros((res, res))
    for i in range(1, res-1):
        for j in range(1, res-1):
            curl[i, j] = (vy[i+1, j] - vy[i-1, j]) / (2 * h) - \
                         (vx[i, j+1] - vx[i, j-1]) / (2 * h)
    return curl


@njit(cache=True)
def gradient_2d_numba(f, h):
    """Central difference gradient."""
    res = f.shape[0]
    gx = np.zeros((res, res))
    gy = np.zeros((res, res))
    for i in range(1, res-1):
        for j in range(res):
            gx[i, j] = (f[i+1, j] - f[i-1, j]) / (2 * h)
    for i in range(res):
        for j in range(1, res-1):
            gy[i, j] = (f[i, j+1] - f[i, j-1]) / (2 * h)
    return gx, gy
