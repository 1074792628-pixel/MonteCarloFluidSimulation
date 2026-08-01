import math
from .types3d import Vec3, RNG3D
from .grid3d import Grid3D

class BiotSavart3D:
    @staticmethod
    def kernel(x: Vec3, y: Vec3) -> Vec3:
        r = y - x
        r2 = r.norm2() + 1e-12
        # 3D Biot-Savart 核: (y-x) / (4π|r|^3)
        factor = 1.0 / (4.0 * math.pi * r2 * math.sqrt(r2))
        return Vec3(r.x * factor, r.y * factor, r.z * factor)
    
    @staticmethod
    def estimate_velocity(vort_grid: Grid3D, x: Vec3, nsamples: int, rng: RNG3D) -> Vec3:
        """在域内均匀采样估计速度（无边界）"""
        # 域范围（从网格信息获取）
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        dx = vort_grid.dx
        ox, oy, oz = vort_grid.ox, vort_grid.oy, vort_grid.oz
        total = Vec3()
        pdf = 1.0 / (nx*ny*nz * dx**3)   # 均匀PDF
        for _ in range(nsamples):
            y = Vec3(rng.uniform(ox, ox+nx*dx),
                     rng.uniform(oy, oy+ny*dx),
                     rng.uniform(oz, oz+nz*dx))
            w = vort_grid.get_vort_interp(y)   # 插值涡量
            k = BiotSavart3D.kernel(x, y)
            # 向量叉积: ω × G (注意方向)
            cross = w.cross(k)
            total = total + cross * (1.0/pdf)
        return total * (1.0/nsamples)