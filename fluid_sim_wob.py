"""
Walk-on-Boundary (WoB) velocity-based fluid simulation.
Based on: Sugimoto et al. 2024 "Velocity-Based Monte Carlo Fluids"

Uses WoB Poisson solver for the projection step (∇²p = ∇·u, p=0 on ∂Ω)
instead of the WoS approach. The WoB method traces random rays to the
boundary and uses the free-space Green's function as a particular solution.

This is a Python/Numba implementation for 2D rectangular domains.
"""
import numpy as np
import time
from numba_wos import (wob_project_grid_numba, wos_diffuse_grid_numba,
                        bilinear_interp, divergence_2d_numba,
                        curl_2d_numba, gradient_2d_numba)
from scipy.ndimage import gaussian_filter as gs


class FluidSimWoB:
    def __init__(self, grid_res=16, nu=0.05, dt=0.03, n_particular=256,
                 n_rays=64, smooth_sigma=0.3):
        self.res = grid_res
        self.nu = nu
        self.dt = dt
        self.n_part = n_particular
        self.n_rays = n_rays
        self.smooth_sigma = smooth_sigma
        self.h = 2.0 / (grid_res - 1)
        xs = np.linspace(-1, 1, grid_res)
        self.X, self.Y = np.meshgrid(xs, xs, indexing='ij')
        self.velocity = np.zeros((grid_res, grid_res, 2))
        self.time = 0.0
        self.stats = {"steps": 0, "total_time": 0.0}

    def set_velocity(self, v):
        self.velocity = v.copy()

    def step(self):
        t0 = time.time()
        res = self.res
        h = self.h

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

        # 2. WoS diffusion (same as other methods)
        du, dv = wos_diffuse_grid_numba(
            self.X, self.Y, u_adv[..., 0], u_adv[..., 1], res,
            self.dt, self.nu, n_samples=max(64, self.n_rays))
        u_diff = np.stack([du, dv], axis=-1)
        for c in range(2):
            u_diff[..., c] = gs(u_diff[..., c], sigma=self.smooth_sigma,
                                 mode='reflect')
        self._enforce_noslip(u_diff)

        # 3. WoB projection: solve ∇²p = ∇·u via WoB, then u_proj = u - ∇p
        p_raw, _ = wob_project_grid_numba(
            self.X, self.Y, u_diff[..., 0], u_diff[..., 1], res, h,
            n_particular=self.n_part, n_rays=self.n_rays)
        p_smooth = gs(p_raw, sigma=self.smooth_sigma, mode='reflect')
        px, py = np.gradient(p_smooth, h, h)
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
                if abs(self.X[i,j]) >= 1.0 or abs(self.Y[i,j]) >= 1.0:
                    vel[i, j] = 0.0

    def vorticity(self):
        return curl_2d_numba(self.velocity[..., 0], self.velocity[..., 1], self.h)

    def kinetic_energy(self):
        return float(0.5 * np.sum(self.velocity**2))

    def divergence_error(self):
        div = divergence_2d_numba(self.velocity[..., 0], self.velocity[..., 1], self.h)
        return float(np.mean(np.abs(div[1:-1, 1:-1]))) if self.res > 2 else 0.0
