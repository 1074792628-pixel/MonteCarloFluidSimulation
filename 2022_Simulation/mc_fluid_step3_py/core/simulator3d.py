import math
import numpy as np
from .types3d import Vec3, RNG3D
from .grid3d import Grid3D
from .biot_savart3d import BiotSavart3D

class Simulator3D:
    def __init__(self, nx, ny, nz, dt, nu=0.0):
        self.nx = nx; self.ny = ny; self.nz = nz
        self.dt = dt
        self.nu = nu                     # 粘性系数
        self.dx = 2.0 / max(nx, ny, nz)  # 适合域 [-1,1]^3
        self.ox = self.oy = self.oz = -1.0
        self.grid = [Grid3D(nx,ny,nz,self.dx,self.ox,self.oy,self.oz) for _ in range(2)]
        self.cur = 0
        self.nmc = 8          # 速度估计采样数
        self.nd = 4           # 扩散样本数（每个点）
        self.h = self.dx      # 拉伸分段长度
        self.rng = RNG3D()
        self._init_vorticity()
        self.dx = 2.0 / max(nx, ny, nz)
    
    def _init_vorticity(self):
        # 初始化两个三维涡环（类似涡环碰撞，简化：两个球状涡旋包，涡量垂直于径向）
        sigma = 0.3
        centers = [Vec3(-0.4, 0.0, 0.0), Vec3(0.4, 0.0, 0.0)]
        strength = 0.8
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    p = self.grid[0].grid_pos(i,j,k)
                    w = Vec3()
                    for c in centers:
                        r = p - c
                        dist = r.norm()
                        envelope = math.exp(-dist*dist/(2*sigma*sigma))
                        # 涡量方向垂直于径向（在yz平面内，产生环向流）
                        # 选择一个垂直方向：例如 (0, -r.z, r.y) 归一化？
                        perp = Vec3(0.0, -r.z, r.y)
                        if perp.norm() > 1e-8:
                            perp = perp * (1.0/perp.norm())
                        w = w + perp * (strength * envelope)
                    self.grid[0].set_vort(i,j,k, w)
                    self.grid[1].set_vort(i,j,k, w)
    
    def step(self):
        prev = self.grid[self.cur]
        nxt = self.grid[1-self.cur]
        nx, ny, nz = self.nx, self.ny, self.nz
        
        print("  计算速度场...")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = prev.grid_pos(i,j,k)
                    v = BiotSavart3D.estimate_velocity(prev, x, self.nmc, self.rng)
                    # 🔒 速度限幅
                    max_vel = 2.0
                    if v.norm() > max_vel:
                        v = v * (max_vel / v.norm())
                    prev.set_vel(i,j,k, v)
        
        print("  更新涡量场...")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = prev.grid_pos(i,j,k)
                    v = prev.get_vel(i,j,k)
                    
                    # 半拉格朗日后向追踪
                    xb = x - v * self.dt
                    
                    # 🔒 限制后向位置在域内
                    margin = self.dx
                    xb.x = max(self.ox + margin, min(xb.x, self.ox + self.nx*self.dx - margin))
                    xb.y = max(self.oy + margin, min(xb.y, self.oy + self.ny*self.dx - margin))
                    xb.z = max(self.oz + margin, min(xb.z, self.oz + self.nz*self.dx - margin))
                    
                    w_avg = Vec3()
                    for _ in range(self.nd):
                        xi = Vec3(self.rng.gaussian(), self.rng.gaussian(), self.rng.gaussian())
                        xd = xb + math.sqrt(2*self.nu*self.dt) * xi
                        
                        # 🔒 限制扩散点也在域内
                        xd.x = max(self.ox + margin, min(xd.x, self.ox + self.nx*self.dx - margin))
                        xd.y = max(self.oy + margin, min(xd.y, self.oy + self.ny*self.dx - margin))
                        xd.z = max(self.oz + margin, min(xd.z, self.oz + self.nz*self.dx - margin))
                        
                        w_prev = prev.get_vort_interp(xd)
                        stretch = self._compute_stretch(prev, xd, w_prev)
                        w_avg = w_avg + (w_prev + self.dt * stretch)
                    
                    w_avg = w_avg * (1.0/self.nd)
                    
                    # 🔒 涡量限幅
                    max_vort = 5.0
                    if w_avg.norm() > max_vort:
                        w_avg = w_avg * (max_vort / w_avg.norm())
                    
                    nxt.set_vort(i,j,k, w_avg)
        
        self.cur = 1 - self.cur
    
    def _compute_stretch(self, grid, x, w):
        h = self.h
        norm_w = w.norm()
        if norm_w < 1e-12:
            return Vec3()
        dir_w = w * (1.0/norm_w)
        x_plus = x + dir_w * (h/2)
        x_minus = x - dir_w * (h/2)
        
        # 🔒 限制拉伸取样点也在域内
        margin = self.dx
        for p in [x_plus, x_minus]:
            p.x = max(self.ox + margin, min(p.x, self.ox + self.nx*self.dx - margin))
            p.y = max(self.oy + margin, min(p.y, self.oy + self.ny*self.dx - margin))
            p.z = max(self.oz + margin, min(p.z, self.oz + self.nz*self.dx - margin))
        
        v_plus = grid.get_vel_interp(x_plus)
        v_minus = grid.get_vel_interp(x_minus)
        diff = (v_plus - v_minus) * (norm_w / h)
        
        # 🔒 拉伸项限幅
        max_stretch = 10.0
        if diff.norm() > max_stretch:
            diff = diff * (max_stretch / diff.norm())
        return diff
    
    @property
    def vorticity_slice(self):
        """返回z=0切面的涡量幅值"""
        k0 = self.nz // 2
        return np.sqrt(self.grid[self.cur].vort[k0,:,:,0]**2 +
                       self.grid[self.cur].vort[k0,:,:,1]**2 +
                       self.grid[self.cur].vort[k0,:,:,2]**2)