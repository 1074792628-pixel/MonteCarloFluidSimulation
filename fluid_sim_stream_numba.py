"""
Stream function projection (for comparison with pressure-Poisson)
Solves: nabla^2 psi = -omega, u = curl psi  (automatically divergence-free)
"""
import numpy as np
import time
from numba_wos import (wos_poisson_grid_numba, wos_diffuse_grid_numba,
                        divergence_2d_numba, gradient_2d_numba,
                        curl_2d_numba, bilinear_interp,
                        boundary_dist_rect_obs, boundary_dist_rect_only)
from scipy.ndimage import gaussian_filter as gs


class FluidSimStream:
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
        res = self.res
        h = self.h

        # 1. Advection
        u_adv = np.zeros_like(self.velocity)
        for i in range(res):
            for j in range(res):
                x, y = self.X[i, j], self.Y[i, j]
                vu = bilinear_interp(x, y, self.velocity[..., 0], res)
                vv = bilinear_interp(x, y, self.velocity[..., 1], res)
                xb, yb = x - vu * self.dt, y - vv * self.dt
                u_adv[i, j, 0] = bilinear_interp(xb, yb, self.velocity[..., 0], res)
                u_adv[i, j, 1] = bilinear_interp(xb, yb, self.velocity[..., 1], res)

        # 2. Numba WoS diffusion
        du, dv = wos_diffuse_grid_numba(
            self.X, self.Y, u_adv[..., 0], u_adv[..., 1], res,
            self.dt, self.nu, n_samples=max(64, self.n_walks//2))
        u_diff = np.stack([du, dv], axis=-1)
        for c in range(2):
            u_diff[..., c] = gs(u_diff[..., c], sigma=0.3, mode='reflect')

        for i in range(res):
            for j in range(res):
                d = (boundary_dist_rect_obs(self.X[i,j], self.Y[i,j], self.ox, self.oy, self.orad)
                     if self.has_obs else boundary_dist_rect_only(self.X[i,j], self.Y[i,j]))
                if d < 0:
                    u_diff[i, j] = 0.0

        # 3. Numba WoS stream function Poisson
        omega = curl_2d_numba(u_diff[..., 0], u_diff[..., 1], h)
        omega = gs(omega, sigma=0.3, mode='reflect')

        psi = wos_poisson_grid_numba(
            self.X, self.Y, -omega, res,
            n_walks=self.n_walks, eps=0.02,
            has_obs=self.has_obs, ox=self.ox, oy=self.oy, orad=self.orad,
            use_neumann=False)
        psi = gs(psi, sigma=0.3, mode='reflect')

        dpsi_dy, dpsi_dx = gradient_2d_numba(psi, h)
        u_proj = np.stack([dpsi_dy, -dpsi_dx], axis=-1)

        for i in range(res):
            for j in range(res):
                d = (boundary_dist_rect_obs(self.X[i,j], self.Y[i,j], self.ox, self.oy, self.orad)
                     if self.has_obs else boundary_dist_rect_only(self.X[i,j], self.Y[i,j]))
                if d < 0:
                    u_proj[i, j] = 0.0

        self.velocity = u_proj
        self.time += self.dt
        self.stats["steps"] += 1
        self.stats["total_time"] += time.time() - t0

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
