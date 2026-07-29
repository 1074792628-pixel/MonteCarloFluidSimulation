"""
Approximation of 2024 "Velocity-Based Monte Carlo Fluids"
  Sugimoto, Batty, Hachisuka (SIGGRAPH 2024)

Note: This Python implementation uses WoS pressure-Poisson projection
as a simplified proxy. The actual 2024 method uses Walk-on-Boundary (WoB)
direct velocity projection, which requires:
  1. CDF-based boundary sampling from mesh
  2. Line-intersection boundary walks
  3. Multi-bounce path integrals with alternating signs
  4. Strongly singular ball volume term + boundary correction

These features require the full GPU implementation via:
  https://github.com/rsugimoto/VelMCFluids

The key difference from the 2022 vorticity-based method is:
  - 2022: Vorticity transport + Biot-Savart, with pressure Poisson projection
  - 2024: Velocity-based operator splitting, pressure Poisson + WoB projection
"""

import numpy as np
import time
from numba_wos import (wos_poisson_grid_numba, wos_diffuse_grid_numba,
                        bilinear_interp, divergence_2d_numba, gradient_2d_numba,
                        curl_2d_numba, boundary_dist_rect_obs,
                        boundary_dist_rect_only)
from scipy.ndimage import gaussian_filter as gs


class FluidSim2024WoB:
    """2024 method using pressure Poisson as simplified proxy."""

    def __init__(self, grid_res=16, nu=0.05, dt=0.03, n_walks=256,
                 has_obs=False, ox=0.0, oy=0.0, orad=0.15):
        self.res = grid_res
        self.nu = nu
        self.dt = dt
        self.n_walks = n_walks
        self.h = 2.0 / (grid_res - 1)
        xs = np.linspace(-1, 1, grid_res)
        self.X, self.Y = np.meshgrid(xs, xs, indexing='ij')
        self.velocity = np.zeros((grid_res, grid_res, 2))
        self.time = 0.0
        self.has_obs = has_obs
        self.ox, self.oy, self.orad = ox, oy, orad
        self.stats = {"steps": 0, "total_time": 0.0}

    def set_velocity(self, v):
        self.velocity = v.copy()

    def step(self):
        t0 = time.time()
        res = self.res; h = self.h

        # 1. Semi-Lagrangian advection
        u_adv = np.zeros_like(self.velocity)
        for i in range(res):
            for j in range(res):
                x, y = self.X[i, j], self.Y[i, j]
                vu = bilinear_interp(x, y, self.velocity[..., 0], res)
                vv = bilinear_interp(x, y, self.velocity[..., 1], res)
                xb, yb = x - vu * self.dt, y - vv * self.dt
                u_adv[i, j, 0] = bilinear_interp(xb, yb, self.velocity[..., 0], res)
                u_adv[i, j, 1] = bilinear_interp(xb, yb, self.velocity[..., 1], res)

        # 2. WoS diffusion
        du, dv = wos_diffuse_grid_numba(
            self.X, self.Y, u_adv[..., 0], u_adv[..., 1], res,
            self.dt, self.nu, n_samples=max(64, self.n_walks//2))
        u_diff = np.stack([du, dv], axis=-1)
        for c in range(2):
            u_diff[..., c] = gs(u_diff[..., c], sigma=0.3, mode='reflect')
        self._enforce_noslip(u_diff)

        # 3. Pressure Poisson projection (simplified proxy for WoB)
        div = divergence_2d_numba(u_diff[..., 0], u_diff[..., 1], h)
        div = gs(div, sigma=0.3, mode='reflect')

        pressure = wos_poisson_grid_numba(
            self.X, self.Y, div, res,
            n_walks=self.n_walks, eps=0.02,
            has_obs=self.has_obs, ox=self.ox, oy=self.oy, orad=self.orad,
            use_neumann=False)
        pressure = gs(pressure, sigma=0.3, mode='reflect')

        px, py = gradient_2d_numba(pressure, h)
        u_diff[..., 0] -= px
        u_diff[..., 1] -= py
        self._enforce_noslip(u_diff)

        self.velocity = u_diff
        self.time += self.dt
        self.stats["steps"] += 1
        self.stats["total_time"] += time.time() - t0

    def _enforce_noslip(self, vel):
        for i in range(self.res):
            for j in range(self.res):
                d = (boundary_dist_rect_obs(self.X[i,j], self.Y[i,j], self.ox, self.oy, self.orad)
                     if self.has_obs else boundary_dist_rect_only(self.X[i,j], self.Y[i,j]))
                if d < 0:
                    vel[i, j] = 0.0

    def vorticity(self):
        return curl_2d_numba(self.velocity[..., 0], self.velocity[..., 1], self.h)

    def kinetic_energy(self):
        return float(0.5 * np.sum(self.velocity**2))

    def divergence_error(self):
        div = divergence_2d_numba(self.velocity[..., 0], self.velocity[..., 1], self.h)
        mask = np.zeros((self.res, self.res), dtype=bool)
        for i in range(self.res):
            for j in range(self.res):
                d = (boundary_dist_rect_obs(self.X[i,j], self.Y[i,j], self.ox, self.oy, self.orad)
                     if self.has_obs else boundary_dist_rect_only(self.X[i,j], self.Y[i,j]))
                mask[i, j] = d > 0
        return float(np.mean(np.abs(div[mask]))) if mask.any() else 0.0
