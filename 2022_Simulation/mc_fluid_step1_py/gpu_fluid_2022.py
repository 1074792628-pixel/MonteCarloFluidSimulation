"""
2022 vorticity method - GPU (CuPy) implementation.
GPU-accelerates the Biot-Savart velocity reconstruction, which is
the computational hotspot (embarrassingly parallel).

Solves 2D inviscid vorticity transport:
  dw/dt + u.dw = 0
  u = Biot-Savart(w)
"""
import numpy as np
import cupy as cp
import time


class GridGPU:
    """Cell-centered grid on [-1,1]^2 storing vorticity."""
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.dx = 2.0 / nx
        self.ox, self.oy = -1.0, -1.0
        self.vort = cp.zeros((ny, nx), dtype=cp.float32)

    def grid_pos(self, i, j):
        return self.ox + (i + 0.5) * self.dx, self.oy + (j + 0.5) * self.dx


@cp.fuse()
def _bilinear_core(fx, fy, v00, v10, v01, v11):
    tx = fx - cp.floor(fx)
    ty = fy - cp.floor(fy)
    return ((1 - ty) * ((1 - tx) * v00 + tx * v10) +
            ty * ((1 - tx) * v01 + tx * v11))


def bilinear_interp_gpu(pos, vort, nx, ny, ox, oy, dx):
    """pos: (P,2) array of query points on GPU. Returns (P,) interpolated vorticity."""
    fx = (pos[:, 0] - ox) / dx - 0.5
    fy = (pos[:, 1] - oy) / dx - 0.5
    i0 = cp.floor(fx).astype(cp.int32)
    j0 = cp.floor(fy).astype(cp.int32)
    i0 = cp.clip(i0, 0, nx - 2)
    j0 = cp.clip(j0, 0, ny - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    v00 = vort[j0, i0]
    v10 = vort[j0, i1]
    v01 = vort[j1, i0]
    v11 = vort[j1, i1]
    return _bilinear_core(fx, fy, v00, v10, v01, v11)


def biot_savart_gpu(query_pos, vort, nx, ny, ox, oy, dx, n_samples, chunk=8192,
                    sample_points=None):
    """GPU Biot-Savart velocity reconstruction for all query points.
    query_pos: (P,2) on GPU. Returns (P,2) velocity.
    Uniform sampling over [-1,1]^2. sample_points: optional fixed (N,2) samples."""
    area = 4.0
    pdf = 1.0 / area
    inv_2pi = 1.0 / (2.0 * np.pi)
    P = query_pos.shape[0]
    result = cp.zeros((P, 2), dtype=cp.float32)

    # Sample points once per call
    if sample_points is None:
        Y = cp.random.uniform(-1.0, 1.0, (n_samples, 2), dtype=cp.float32)
    else:
        Y = sample_points
    w = bilinear_interp_gpu(Y, vort, nx, ny, ox, oy, dx)  # (N,)
    wpdf = (w * (1.0 / pdf))[None, :]  # (1, N)

    for start in range(0, P, chunk):
        end = min(start + chunk, P)
        X = query_pos[start:end]  # (chunk, 2)

        dxm = X[:, None, 0] - Y[None, :, 0]  # (chunk, N)
        dym = X[:, None, 1] - Y[None, :, 1]
        r2 = dxm * dxm + dym * dym + 1e-12
        bx = dym * inv_2pi / r2
        by = -dxm * inv_2pi / r2
        result[start:end, 0] = cp.sum(bx * wpdf, axis=1) / n_samples
        result[start:end, 1] = cp.sum(by * wpdf, axis=1) / n_samples
    return result


class Sim2022GPU:
    """2D viscous vorticity transport simulator, GPU-accelerated.
    Feynman-Kac diffusion: w_new(x) = E[w(x + sqrt(2 nu dt) Z)]. Inviscid if nu=0."""
    def __init__(self, nx, ny, dt, nmc, nu=0.0, n_diffuse=128):
        self.nx, self.ny, self.dt, self.nmc, self.nu, self.n_diffuse = nx, ny, dt, nmc, nu, n_diffuse
        self.dx = 2.0 / nx
        self.ox, self.oy = -1.0, -1.0
        self.grid = [GridGPU(nx, ny), GridGPU(nx, ny)]
        self.cur = 0
        self._init_vorticity()
        # Precompute all grid query positions once (P, 2)
        xs = (np.arange(nx) + 0.5) * self.dx + self.ox
        ys = (np.arange(ny) + 0.5) * self.dx + self.oy
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        self.query_pos = cp.asarray(np.stack([XX, YY], axis=-1).reshape(-1, 2), dtype=cp.float32)
        # Fixed deterministic sample points for reproducible Biot-Savart integration
        rng = np.random.default_rng(42)
        self.sample_points = cp.asarray(rng.uniform(-1.0, 1.0, (self.nmc, 2)), dtype=cp.float32)

    def _init_vorticity(self):
        c1 = (-0.7, 1.0 / 6.0)
        c2 = (-0.7, -1.0 / 6.0)
        radius = 0.8 / 6.0
        for idx, g in enumerate(self.grid):
            for j in range(self.ny):
                for i in range(self.nx):
                    px, py = g.grid_pos(i, j)
                    d1 = np.hypot(px - c1[0], py - c1[1])
                    d2 = np.hypot(px - c2[0], py - c2[1])
                    w = (1.0 if d1 <= radius else 0.0) - (1.0 if d2 <= radius else 0.0)
                    g.vort[j, i] = w

    def _init_taylor_green(self):
        """Alternative: Taylor-Green vorticity 2*pi*cos(pi x)cos(pi y)."""
        for g in self.grid:
            for j in range(self.ny):
                for i in range(self.nx):
                    px, py = g.grid_pos(i, j)
                    g.vort[j, i] = 2 * np.pi * np.cos(np.pi * px) * np.cos(np.pi * py)

    def step(self):
        prev = self.grid[self.cur]
        nxt = self.grid[1 - self.cur]
        nxt.vort[...] = 0.0
        # Velocity at all grid points via GPU Biot-Savart
        vel = biot_savart_gpu(self.query_pos, prev.vort, self.nx, self.ny,
                              self.ox, self.oy, self.dx, self.nmc,
                              sample_points=self.sample_points)
        vel = vel.reshape(self.nx, self.ny, 2)
        # Semi-Lagrangian backward trace
        xs = self.query_pos[:, 0] - vel.reshape(-1, 2)[:, 0] * self.dt
        ys = self.query_pos[:, 1] - vel.reshape(-1, 2)[:, 1] * self.dt
        # Clamp to domain
        xs = cp.clip(xs, -1.0, 1.0)
        ys = cp.clip(ys, -1.0, 1.0)
        back_pos = cp.stack([xs, ys], axis=-1)
        w_new = bilinear_interp_gpu(back_pos, prev.vort, self.nx, self.ny,
                                    self.ox, self.oy, self.dx)
        # Feynman-Kac diffusion: w_new(x) = E[w_adv(x + sqrt(2 nu dt) Z)]
        if self.nu > 0.0:
            sigma = np.sqrt(2.0 * self.nu * self.dt)
            # Sample perturbations for all grid points
            rng = cp.random.default_rng()
            Z = rng.standard_normal((self.nx * self.ny, 2), dtype=cp.float32) * sigma
            diff_pos = cp.clip(back_pos + Z, -1.0, 1.0)
            w_diff = bilinear_interp_gpu(diff_pos, w_new.reshape(self.nx, self.ny),
                                         self.nx, self.ny, self.ox, self.oy, self.dx)
            w_new = w_diff
        nxt.vort.reshape(-1)[...] = w_new
        self.cur = 1 - self.cur

    @property
    def vorticity(self):
        return cp.asnumpy(self.grid[self.cur].vort)


def run_benchmark(nx=64, ny=64, dt=0.05, nmc=128, steps=40, use_tg=False):
    sim = Sim2022GPU(nx, ny, dt, nmc)
    if use_tg:
        sim._init_taylor_green()
    t0 = time.time()
    for _ in range(steps):
        sim.step()
    elapsed = time.time() - t0
    return sim, elapsed


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=64)
    p.add_argument('--nmc', type=int, default=128)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--taylor_green', action='store_true')
    args = p.parse_args()
    sim, t = run_benchmark(args.nx, args.nx, 0.05, args.nmc, args.steps, args.taylor_green)
    print(f"GPU 2022: {args.nx}x{args.nx}, nmc={args.nmc}, {args.steps} steps, time={t:.2f}s")
