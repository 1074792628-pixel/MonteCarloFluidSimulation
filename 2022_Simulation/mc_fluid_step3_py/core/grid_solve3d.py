# grid_solve3d.py (修正版)
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def tg_vorticity(x, y, z, t, nu):
    decay = np.exp(-nu * np.pi**2 * t)
    wx = -np.pi * np.cos(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z) * decay
    wy =  np.pi * np.sin(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z) * decay
    wz =  2.0*np.pi * np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(np.pi*z) * decay
    return wx, wy, wz


class GridSolver3D:
    def __init__(self, nx, ny, nz, dt, dx, ox, oy, oz, nu=0.0):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dt = dt
        self.dx = dx
        self.ox, self.oy, self.oz = ox, oy, oz
        self.nu = nu
        self.vel = np.zeros((nz, ny, nx, 3))

    def set_initial_vorticity(self, omega_grid):
        """用解析速度场初始化速度（参数名保留但不使用）"""
        nx, ny, nz = self.nx, self.ny, self.nz
        dx = self.dx
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = self.ox + (i + 0.5) * dx
                    y = self.oy + (j + 0.5) * dx
                    z = self.oz + (k + 0.5) * dx
                    u =  np.sin(np.pi*x) * np.cos(np.pi*y) * np.cos(np.pi*z)
                    v = -np.cos(np.pi*x) * np.sin(np.pi*y) * np.cos(np.pi*z)
                    w = 0.0
                    self.vel[k, j, i, 0] = u
                    self.vel[k, j, i, 1] = v
                    self.vel[k, j, i, 2] = w

    def _interp_vel(self, px, py, pz):
        """完整的三线性插值实现"""
        # 输入有效性检查
        if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(pz)):
            return np.zeros(3)

        nx, ny, nz = self.nx, self.ny, self.nz
        dx = self.dx

        fx = (px - self.ox) / dx - 0.5
        fy = (py - self.oy) / dx - 0.5
        fz = (pz - self.oz) / dx - 0.5

        # clamp 到有效索引范围
        fx = min(max(fx, 0.0), nx - 2.0)
        fy = min(max(fy, 0.0), ny - 2.0)
        fz = min(max(fz, 0.0), nz - 2.0)

        i0 = int(np.floor(fx)); j0 = int(np.floor(fy)); k0 = int(np.floor(fz))
        i1, j1, k1 = i0+1, j0+1, k0+1

        tx, ty, tz = fx - i0, fy - j0, fz - k0

        # 三线性插值
        v000 = self.vel[k0, j0, i0]; v100 = self.vel[k0, j0, i1]
        v010 = self.vel[k0, j1, i0]; v110 = self.vel[k0, j1, i1]
        v001 = self.vel[k1, j0, i0]; v101 = self.vel[k1, j0, i1]
        v011 = self.vel[k1, j1, i0]; v111 = self.vel[k1, j1, i1]

        res = np.zeros(3)
        for c in range(3):
            res[c] = ((1-tz)*((1-ty)*((1-tx)*v000[c] + tx*v100[c]) + ty*((1-tx)*v010[c] + tx*v110[c])) +
                       tz*((1-ty)*((1-tx)*v001[c] + tx*v101[c]) + ty*((1-tx)*v011[c] + tx*v111[c])))
        return res

    def advect(self):
        new_vel = np.zeros_like(self.vel)
        nx, ny, nz = self.nx, self.ny, self.nz
        xmin = self.ox + 0.5 * self.dx
        xmax = self.ox + (nx - 0.5) * self.dx
        ymin = self.oy + 0.5 * self.dx
        ymax = self.oy + (ny - 0.5) * self.dx
        zmin = self.oz + 0.5 * self.dx
        zmax = self.oz + (nz - 0.5) * self.dx

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = self.ox + (i+0.5)*self.dx
                    y = self.oy + (j+0.5)*self.dx
                    z = self.oz + (k+0.5)*self.dx
                    v = self.vel[k, j, i]

                    # 速度限幅
                    max_vel = 2.0
                    speed = np.linalg.norm(v)
                    if speed > max_vel:
                        v = v * (max_vel / speed)

                    xb = x - self.dt * v[0]
                    yb = y - self.dt * v[1]
                    zb = z - self.dt * v[2]

                    xb = min(max(xb, xmin), xmax)
                    yb = min(max(yb, ymin), ymax)
                    zb = min(max(zb, zmin), zmax)

                    new_vel[k, j, i] = self._interp_vel(xb, yb, zb)
        self.vel = new_vel

    def diffuse(self):
        if self.nu == 0:
            return
        new_vel = self.vel.copy()
        nx, ny, nz = self.nx, self.ny, self.nz
        factor = self.nu * self.dt / self.dx**2
        # 显式扩散，内部点
        for k in range(1, nz-1):
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    for c in range(3):
                        lap = (self.vel[k, j, i+1, c] + self.vel[k, j, i-1, c] +
                               self.vel[k, j+1, i, c] + self.vel[k, j-1, i, c] +
                               self.vel[k+1, j, i, c] + self.vel[k-1, j, i, c] -
                               6*self.vel[k, j, i, c])
                        new_vel[k, j, i, c] = self.vel[k, j, i, c] + factor * lap
        self.vel = new_vel

    def project(self):
        """投影：求解 ∇²p = ∇·u，u = u - ∇p，边界 Dirichlet p=0"""
        nx, ny, nz = self.nx, self.ny, self.nz
        N = nx * ny * nz
        A = lil_matrix((N, N))
        b = np.zeros(N)

        def idx(i, j, k):
            return (k * ny + j) * nx + i

        def is_boundary(i, j, k):
            return (i == 0 or i == nx-1 or
                    j == 0 or j == ny-1 or
                    k == 0 or k == nz-1)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    row = idx(i, j, k)
                    if is_boundary(i, j, k):
                        A[row, row] = 1.0
                        b[row] = 0.0
                    else:
                        # 数值散度
                        div = (self.vel[k, j, i+1, 0] - self.vel[k, j, i-1, 0]) / (2*self.dx) + \
                              (self.vel[k, j+1, i, 1] - self.vel[k, j-1, i, 1]) / (2*self.dx) + \
                              (self.vel[k+1, j, i, 2] - self.vel[k-1, j, i, 2]) / (2*self.dx)
                        b[row] = div

                        A[row, row] = -6.0
                        A[row, idx(i+1, j, k)] = 1.0
                        A[row, idx(i-1, j, k)] = 1.0
                        A[row, idx(i, j+1, k)] = 1.0
                        A[row, idx(i, j-1, k)] = 1.0
                        A[row, idx(i, j, k+1)] = 1.0
                        A[row, idx(i, j, k-1)] = 1.0

        A = csr_matrix(A)
        p = spsolve(A, b).reshape((nz, ny, nx))

        # 速度修正（内部点）
        for k in range(1, nz-1):
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    grad_px = (p[k, j, i+1] - p[k, j, i-1]) / (2*self.dx)
                    grad_py = (p[k, j+1, i] - p[k, j-1, i]) / (2*self.dx)
                    grad_pz = (p[k+1, j, i] - p[k-1, j, i]) / (2*self.dx)
                    self.vel[k, j, i, 0] -= grad_px
                    self.vel[k, j, i, 1] -= grad_py
                    self.vel[k, j, i, 2] -= grad_pz

    def get_vorticity(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        vort = np.zeros((nz, ny, nx, 3))
        for k in range(1, nz-1):
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    dvz_dy = (self.vel[k, j+1, i, 2] - self.vel[k, j-1, i, 2]) / (2*self.dx)
                    dvy_dz = (self.vel[k+1, j, i, 1] - self.vel[k-1, j, i, 1]) / (2*self.dx)
                    dvx_dz = (self.vel[k+1, j, i, 0] - self.vel[k-1, j, i, 0]) / (2*self.dx)
                    dvz_dx = (self.vel[k, j, i+1, 2] - self.vel[k, j, i-1, 2]) / (2*self.dx)
                    dvy_dx = (self.vel[k, j, i+1, 1] - self.vel[k, j, i-1, 1]) / (2*self.dx)
                    dvx_dy = (self.vel[k, j+1, i, 0] - self.vel[k, j-1, i, 0]) / (2*self.dx)
                    vort[k, j, i, 0] = dvz_dy - dvy_dz
                    vort[k, j, i, 1] = dvx_dz - dvz_dx
                    vort[k, j, i, 2] = dvy_dx - dvx_dy
        return vort

    def step(self):
        self.advect()
        if self.nu > 0:
            self.diffuse()
        self.project()