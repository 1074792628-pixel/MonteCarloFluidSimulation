import numpy as np
from .types import Vec2, RNG
from .grid import Grid
from .geometry import Geometry
from numba import njit
import math

# 泊松格林函数及其梯度（2D 球）
# 对于半径为 R 的球，中心在 x，内部点 y，r = |y-x|
# 格林函数 G(R, r) = (1/(2π)) log(R/r)
# 梯度 ∇G(R, r) = - (y-x) / (2π r²) (因为函数仅依赖 r)

EPSILON = 1e-4
MAX_STEPS = 200

@njit
def _numba_solve_psi_walk(x0, nx, ny, dx, ox, oy, 
                           vort_data, obstacle_data, eps, max_steps):
    """
    Numba 编译的单条 WoS 路径估计 Ψ(x)。
    obstacle_data 为障碍物的参数（简化为圆形：(cx,cy,r) 或数组）
    vort_data: (ny, nx) 的涡量场
    """
    total = 0.0
    cx, cy, r = obstacle_data[0], obstacle_data[1], obstacle_data[2]
    
    x, y = x0[0], x0[1]
    for _ in range(max_steps):
        # 到边界和障碍物的距离
        d_out = min(x - ox, (ox + nx*dx) - x, y - oy, (oy + ny*dx) - y)
        d_obs = math.sqrt((x-cx)**2 + (y-cy)**2) - r
        R = min(d_out, d_obs)
        if R < eps:
            break
        # 球内采样 y（均匀径向分布）
        rho = R * math.sqrt(np.random.uniform(0,1))
        theta = np.random.uniform(0, 2*math.pi)
        yx = x + rho * math.cos(theta)
        yy = y + rho * math.sin(theta)
        # 插值涡量
        w = _bilinear_interp(vort_data, yx, yy, nx, ny, dx, ox, oy)
        # 格林函数贡献
        G = (1.0/(2*math.pi)) * math.log(max(R/rho, 1e-12))
        vol = math.pi * R * R
        total += vol * w * G
        # 球面采样下一步位置
        theta2 = np.random.uniform(0, 2*math.pi)
        x = x + R * math.cos(theta2)
        y = y + R * math.sin(theta2)
    return total

@njit
def _bilinear_interp(data, px, py, nx, ny, dx, ox, oy):
    """双线性插值（Numba 兼容）"""
    fx = (px - ox) / dx - 0.5
    fy = (py - oy) / dx - 0.5
    i0 = int(math.floor(fx))
    j0 = int(math.floor(fy))
    i1 = i0 + 1; j1 = j0 + 1
    i0 = max(0, min(i0, nx-2)); i1 = i0+1
    j0 = max(0, min(j0, ny-2)); j1 = j0+1
    tx = fx - i0; ty = fy - j0
    v00 = data[j0, i0]; v10 = data[j0, i1]
    v01 = data[j1, i0]; v11 = data[j1, i1]
    return (1-ty)*((1-tx)*v00 + tx*v10) + ty*((1-tx)*v01 + tx*v11)

class WoS_solver:
    def __init__(self, geometry: Geometry, grid: Grid):
        self.geo = geometry
        self.grid = grid
        self.rng = RNG()
    
    def _sample_sphere_dir(self) -> Vec2:
        """ 均匀采样单位圆上的方向 """
        theta = self.rng.uniform(0, 2*np.pi)
        return Vec2(np.cos(theta), np.sin(theta))
    
    def _solve_psi_walk(self, x: Vec2) -> float:
        obstacle_data = np.array([0.0, 0.0, 0.3])  # 圆心、半径
        return _numba_solve_psi_walk(
            np.array([x.x, x.y]),
            self.grid.nx, self.grid.ny, self.grid.dx,
            self.grid.ox, self.grid.oy,
            self.grid.vort,     # NumPy 数组
            obstacle_data,
            EPSILON, MAX_STEPS
        )
    
    def estimate_psi_grad(self, x: Vec2, npaths: int = 100):
        """ 使用梯度 WoS 估计 ∇Ψ(x) （返回 Vec2） """
        grad_sum = Vec2()
        for _ in range(npaths):
            R0 = self.geo.distance(x)
            if R0 < EPSILON:
                continue
            # 第一球的梯度边界项（对偶采样）
            dir = self._sample_sphere_dir()
            x1 = x + dir * R0
            x1_anti = x - dir * R0   # 对侧点
            
            psi1 = self._solve_psi_walk(x1)
            psi1_anti = self._solve_psi_walk(x1_anti)
            
            # 边界梯度贡献（2D: n=2）
            n = 2
            grad = Vec2()
            grad = grad + (n / (2*R0)) * psi1 * dir
            grad = grad + (n / (2*R0)) * psi1_anti * (-dir)   # 注意法向与 dir 相反
            
            # 第一球的源项梯度贡献
            # 在球内采样 y，贡献 vol * w * ∇G
            r_samp = R0 * np.sqrt(self.rng.uniform(0, 1))
            theta_samp = self.rng.uniform(0, 2*np.pi)
            y = x + Vec2(r_samp*np.cos(theta_samp), r_samp*np.sin(theta_samp))
            w = self.grid.get_vort(y)
            # ∇G(x,y) = - (y-x) / (2π r²)
            r_vec = y - x
            r2 = r_vec.norm2()
            if r2 > 1e-12:
                grad_G = r_vec * (-1.0/(2*np.pi*r2))
            else:
                grad_G = Vec2()
            vol = np.pi * R0*R0
            grad = grad + vol * w * grad_G
            
            grad_sum = grad_sum + grad
        return grad_sum * (1.0 / npaths)
    
    def velocity_at(self, x: Vec2, npaths: int = 100) -> Vec2:
        """ 通过流函数梯度恢复速度 v = (-∂Ψ/∂y, ∂Ψ/∂x) """
        if not self.geo.inside_domain(x):
            return Vec2()
        grad_psi = self.estimate_psi_grad(x, npaths)
        # 旋度：v = -∇×Ψ = (-∂Ψ/∂y, ∂Ψ/∂x)
        return Vec2(-grad_psi.y, grad_psi.x)