"""
2022 Paper: "A Monte Carlo Method for Fluid Simulation"
  Rioux-Lavoie, Sugimoto, et al. (SIGGRAPH 2022)

Operator splitting with WoS for diffusion and pressure projection:
  1. Semi-Lagrangian advection
  2. WoS diffusion: ∂u/∂t = ν∇²u
  3. Pressure Poisson: ∇²p = ∇·u*  →  u = u* - ∇p
"""

import numpy as np
import time
from wos_core import (wos_poisson_2d, divergence_2d, gradient_2d, curl_2d,
                       semi_lagrangian_advect)


class FluidSim2022:
    def __init__(self, domain, grid_res=32, nu=0.1, dt=0.05, n_walks=128):
        self.domain = domain
        self.res = grid_res
        self.nu = nu
        self.dt = dt
        self.n_walks = n_walks
        self.h = 2.0 / (grid_res - 1)
        xs = np.linspace(-1, 1, grid_res)
        self.X, self.Y = np.meshgrid(xs, xs, indexing='ij')
        self.velocity = np.zeros((grid_res, grid_res, 2))
        self.time = 0.0
        self.stats = {"pressure_solves": 0, "diffusion_steps": 0, "total_time": 0.0}
        self.pts = np.stack([self.X, self.Y], axis=-1)

    def set_velocity(self, v):
        self.velocity = v.copy()

    def step(self):
        t0 = time.time()
        u = self.velocity.copy()

        # 1. Semi-Lagrangian advection
        u_adv = semi_lagrangian_advect(u, u, self.dt)

        # 2. WoS diffusion (each velocity component)
        def field_at(pts):
            out = np.zeros((len(pts), 2))
            for k, pt in enumerate(pts):
                out[k] = self._interp_vec(pt, u_adv)
            return out

        from wos_core import wos_diffuse_2d, gaussian_smooth
        u_diff = wos_diffuse_2d(self.pts, field_at, self.domain,
                                 self.dt, self.nu, n_samples=self.n_walks)
        u_diff = gaussian_smooth(u_diff, sigma=0.4)
        self.stats["diffusion_steps"] += 1

        # 3. Pressure projection via WoS Poisson
        div = divergence_2d(u_diff[..., 0], u_diff[..., 1], self.h)

        def src(p):
            return np.array([self._interp(pt, div) for pt in p])

        pressure = wos_poisson_2d(self.pts, src, self.domain,
                                   n_walks=self.n_walks, neumann_bc=False)
        pressure = gaussian_smooth(pressure, sigma=0.5)
        self.stats["pressure_solves"] += 1

        # u = u* - ∇p
        px, py = gradient_2d(pressure, self.h)
        u_diff[..., 0] -= px
        u_diff[..., 1] -= py

        # Enforce no-slip at boundaries
        self._enforce_noslip(u_diff)
        self.velocity = u_diff
        self.time += self.dt
        self.stats["total_time"] += time.time() - t0

    def _interp_vec(self, pt, grid):
        return np.array([self._interp(pt, grid[..., 0]),
                         self._interp(pt, grid[..., 1])])

    def _interp(self, pt, grid):
        shape = grid.shape
        ix = (pt[0] + 1.0) * (shape[0] - 1) / 2.0
        iy = (pt[1] + 1.0) * (shape[1] - 1) / 2.0
        i, j = int(np.floor(ix)), int(np.floor(iy))
        fi, fj = ix - i, iy - j
        i = np.clip(i, 0, shape[0] - 2)
        j = np.clip(j, 0, shape[1] - 2)
        return ((1-fi)*(1-fj)*grid[i, j] + fi*(1-fj)*grid[i+1, j] +
                (1-fi)*fj*grid[i, j+1] + fi*fj*grid[i+1, j+1])

    def _enforce_noslip(self, vel):
        mask = ~self.domain.inside(self.pts.reshape(-1, 2)).reshape(self.res, self.res)
        vel[mask] = 0.0

    def vorticity(self):
        return curl_2d(self.velocity[..., 0], self.velocity[..., 1], self.h)

    def kinetic_energy(self):
        return 0.5 * np.sum(self.velocity**2)

    def divergence_error(self):
        div = divergence_2d(self.velocity[..., 0], self.velocity[..., 1], self.h)
        inside = self.domain.inside(self.pts.reshape(-1, 2)).reshape(self.res, self.res)
        return float(np.mean(np.abs(div[inside]))) if inside.any() else 0.0
