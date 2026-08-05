import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from core.types import Vec2
# ============================================================
# 网格流函数法（传统方法，可处理复杂边界）
# ============================================================
class GridStreamFunctionSolver:
    def __init__(self, nx, ny, dt, dx, ox, oy, geometry):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = dx
        self.ox = ox
        self.oy = oy
        self.geometry = geometry
        self.omega = np.zeros((ny, nx))
        self.psi = np.zeros((ny, nx))
        self.vel = np.zeros((ny, nx, 2))
    def set_omega(self, omega_grid):
        self.omega = omega_grid.copy()
    def _inside_mask(self):
        """返回网格点是否在流体域内（不在固体内部且在外边界内）"""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        for j in range(self.ny):
            for i in range(self.nx):
                p = Vec2(self.ox + (i + 0.5) * self.dx,
                         self.oy + (j + 0.5) * self.dx)
                mask[j, i] = self.geometry.inside_domain(p)
        return mask
    def solve_psi(self):
        """有限差分求解 Poisson: ∇²ψ = -ω, 边界ψ=0"""
        nx, ny = self.nx, self.ny
        N = nx * ny
        A = lil_matrix((N, N))
        b = np.zeros(N)
        def idx(i, j):
            return j * nx + i
        for j in range(ny):
            for i in range(nx):
                row = idx(i, j)
                p = Vec2(self.ox + (i + 0.5) * self.dx,
                         self.oy + (j + 0.5) * self.dx)
                inside = self.geometry.inside_domain(p)
                if not inside:
                    # 固体内部或域外：ψ=0
                    A[row, row] = 1.0
                    b[row] = 0.0
                else:
                    # 五点差分 Laplacian
                    A[row, row] = -4.0
                    b[row] = self.omega[j, i] * self.dx**2
                    # 邻居
                    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            A[row, idx(ni, nj)] = 1.0
                        else:
                            # 域外边界：视为ψ=0（Dirichlet），贡献到右边
                            b[row] -= 0.0   # 已默认邻居0
        A = csr_matrix(A)
        self.psi = spsolve(A, b).reshape((ny, nx))
    def compute_velocity(self):
        """从ψ中心差分计算速度 u=(∂ψ/∂y, -∂ψ/∂x)"""
        psi = self.psi
        nx, ny = self.nx, self.ny
        for j in range(ny):
            for i in range(nx):
                jm = max(j-1, 0); jp = min(j+1, ny-1)
                im = max(i-1, 0); ip = min(i+1, nx-1)
                # ∂ψ/∂y
                dpsi_dy = (psi[jp, i] - psi[jm, i]) / (2*self.dx) if jp != jm else 0
                # ∂ψ/∂x
                dpsi_dx = (psi[j, ip] - psi[j, im]) / (2*self.dx) if ip != im else 0
                self.vel[j, i, 0] = dpsi_dy
                self.vel[j, i, 1] = -dpsi_dx
    def _interp(self, field, x, y):
        """双线性插值（带域外/固体处理）"""
        fx = (x - self.ox) / self.dx - 0.5
        fy = (y - self.oy) / self.dx - 0.5
        i0 = int(np.floor(fx)); j0 = int(np.floor(fy))
        tx = fx - i0; ty = fy - j0
        i0 = max(0, min(i0, self.nx-2)); j0 = max(0, min(j0, self.ny-2))
        i1 = i0+1; j1 = j0+1
        return ((1-ty)*((1-tx)*field[j0,i0] + tx*field[j0,i1]) +
                ty*((1-tx)*field[j1,i0] + tx*field[j1,i1]))
    def advect_omega(self):
        """半拉格朗日平流涡量"""
        new_omega = np.zeros_like(self.omega)
        for j in range(self.ny):
            for i in range(self.nx):
                x = self.ox + (i+0.5)*self.dx
                y = self.oy + (j+0.5)*self.dx
                vx = self.vel[j, i, 0]
                vy = self.vel[j, i, 1]
                xb = x - self.dt * vx
                yb = y - self.dt * vy
                # 域内限制
                p_b = Vec2(xb, yb)
                if not self.geometry.inside_domain(p_b):
                    # 若落在固体内，取最近域内点（简化：取原始点）
                    xb, yb = x, y
                new_omega[j, i] = self._interp(self.omega, xb, yb)
        self.omega = new_omega
    def step(self):
        self.solve_psi()
        self.compute_velocity()
        self.advect_omega()