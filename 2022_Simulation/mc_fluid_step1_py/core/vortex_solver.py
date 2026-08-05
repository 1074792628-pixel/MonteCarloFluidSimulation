import numpy as np

class VortexParticleSolver:
    """2D 涡量粒子法（基于 Biot-Savart + 高斯核）"""
    def __init__(self, nx, ny, dt, dx):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = dx
        self.sigma = dx
        self.positions = np.empty((0, 2), dtype=np.float64)   # 初始化为空数组
        self.circulations = np.empty((0,), dtype=np.float64)  # 初始化为空数组
        self.t = 0.0

    def initialize_from_grid(self, omega_grid, ox, oy):
        ny, nx = omega_grid.shape
        xs = ox + (np.arange(nx) + 0.5) * self.dx
        ys = oy + (np.arange(ny) + 0.5) * self.dx
        XX, YY = np.meshgrid(xs, ys)
        positions = np.stack([XX.ravel(), YY.ravel()], axis=1)
        circs = omega_grid.ravel() * self.dx**2
        mask = np.abs(circs) > 1e-10
        self.positions = positions[mask].astype(np.float64)
        self.circulations = circs[mask].astype(np.float64)

    def compute_velocity(self, points):
        """计算 points (M,2) 处的速度（向量化）"""
        N = len(self.positions)          # 现在 self.positions 总是 ndarray，类型无误
        M = len(points)
        if N == 0 or M == 0:
            return np.zeros((M, 2))

        dx = points[:, 0][:, None] - self.positions[:, 0][None, :]  # (M,N)
        dy = points[:, 1][:, None] - self.positions[:, 1][None, :]
        r2 = dx * dx + dy * dy + 1e-12
        sigma2 = self.sigma**2
        kernel = 1.0 / (2 * np.pi * r2) * (1 - np.exp(-r2 / (2 * sigma2)))
        vx = np.sum(-self.circulations[None, :] * dy * kernel, axis=1)
        vy = np.sum(self.circulations[None, :] * dx * kernel, axis=1)
        return np.stack([vx, vy], axis=1)

    def step(self):
        v1 = self.compute_velocity(self.positions)
        mid = self.positions + 0.5 * self.dt * v1
        v2 = self.compute_velocity(mid)
        self.positions += self.dt * v2
        self.t += self.dt

    def get_vorticity_grid(self, nx, ny, ox, oy):
        """将粒子投影回网格（高斯核叠加）"""
        if len(self.positions) == 0:
            return np.zeros((ny, nx))
        xs = ox + (np.arange(nx) + 0.5) * self.dx
        ys = oy + (np.arange(ny) + 0.5) * self.dx
        XX, YY = np.meshgrid(xs, ys)

        PX = self.positions[:, 0][None, None, :]
        PY = self.positions[:, 1][None, None, :]
        dx = XX[:, :, None] - PX
        dy = YY[:, :, None] - PY
        r2 = dx * dx + dy * dy
        w = np.exp(-r2 / (2 * self.sigma**2))
        grid = np.sum(w * self.circulations[None, None, :], axis=2) / (2 * np.pi * self.sigma**2)
        return grid