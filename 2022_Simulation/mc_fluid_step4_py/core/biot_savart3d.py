import math
import numpy as np
from .types3d import Vec3, RNG3D
from .grid3d import Grid3D

class BiotSavart3D:
    @staticmethod
    def kernel(x: Vec3, y: Vec3) -> Vec3:
        r = y - x
        r2 = r.norm2() + 1e-12
        factor = 1.0 / (4.0 * math.pi * r2 * math.sqrt(r2))
        return Vec3(r.x * factor, r.y * factor, r.z * factor)

    @staticmethod
    def kernel_vectorized(x: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        x: (3,) 或 (1, 3)
        y_batch: (N, 3)
        返回: (N, 3)
        """
        r = y_batch - x[np.newaxis, :]           # (N, 3) - (1, 3) = (N, 3)
        r2 = np.sum(r**2, axis=1, keepdims=True) + 1e-12   # (N, 1)
        r_mag = np.sqrt(r2)                                # (N, 1)
        factor = 1.0 / (4.0 * np.pi * r2 * r_mag)          # (N, 1)
        return r * factor                                  # (N, 3) * (N, 1) = (N, 3)

    @staticmethod
    def importance_sample(vort_grid: Grid3D, rng: RNG3D,
                          nsamples: int, x: Vec3) -> list:
        """
        按涡量幅值分布在域内采样（使用CDF方法）。
        返回 (y, pdf) 列表，供估计器使用。
        """
        # 为简化，这里采用预构建涡量幅值CDF的策略（一次性构建）
        # 但实际高效实现应使用别名表或随机索引。
        # 由于篇幅，仅提供示意：
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        dx = vort_grid.dx
        ox, oy, oz = vort_grid.ox, vort_grid.oy, vort_grid.oz

        # 构建权重数组（涡量幅值）
        weights = []
        positions = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    w = vort_grid.get_vort(i,j,k)
                    mag = w.norm()
                    if mag > 1e-12:
                        weights.append(mag)
                        positions.append((i,j,k))
        if not weights:
            # 无涡量——fallback 均匀采样
            return BiotSavart3D.uniform_sample(vort_grid, rng, nsamples, x)
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        cdf = np.cumsum(weights)

        samples = []
        for _ in range(nsamples):
            r = rng.uniform()
            idx = np.searchsorted(cdf, r)
            i,j,k = positions[idx]
            # 在网格单元内均匀抖动实现连续采样
            y = Vec3(
                ox + (i + rng.uniform(0,1)) * dx,
                oy + (j + rng.uniform(0,1)) * dx,
                oz + (k + rng.uniform(0,1)) * dx
            )
            pdf = weights[idx] / (dx**3)  # 密度变换
            samples.append((y, pdf))
        return samples

    @staticmethod
    def uniform_sample(vort_grid: Grid3D, rng: RNG3D,
                       nsamples: int, x: Vec3) -> list:
        """均匀采样（备用）"""
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        dx = vort_grid.dx
        ox, oy, oz = vort_grid.ox, vort_grid.oy, vort_grid.oz
        volume = nx*ny*nz * dx**3
        pdf_uniform = 1.0 / volume
        samples = []
        for _ in range(nsamples):
            y = Vec3(rng.uniform(ox, ox+nx*dx),
                     rng.uniform(oy, oy+ny*dx),
                     rng.uniform(oz, oz+nz*dx))
            samples.append((y, pdf_uniform))
        return samples

    @staticmethod
    def estimate_velocity_vectorized(vort_grid, x_vec3, nsamples, rng):
        """向量化均匀采样估计速度"""
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        dx = vort_grid.dx
        ox, oy, oz = vort_grid.ox, vort_grid.oy, vort_grid.oz
        # 一次性生成所有采样点 (nsamples, 3)
        ys = np.random.uniform(
            [ox, oy, oz],
            [ox + nx*dx, oy + ny*dx, oz + nz*dx],
            size=(nsamples, 3)
        )
        # 插值涡量（需向量化的插值函数，见下方说明）
        ws = vort_grid.get_vort_interp_batch(ys[:,0], ys[:,1], ys[:,2])  # (nsamples,3)
        # 核向量
        x = np.array([x_vec3.x, x_vec3.y, x_vec3.z])   # (3,)
        Gs = BiotSavart3D.kernel_vectorized(x, ys)      # (nsamples,3)
        # 叉积 ω × G
        cross = np.cross(ws, Gs)                        # (nsamples,3)
        # 均匀 PDF = 1/体积
        volume = nx * ny * nz * dx**3
        pdf = 1.0 / volume
        # 平均
        v_mean = np.mean(cross / pdf, axis=0)           # (3,)
        return Vec3(v_mean[0], v_mean[1], v_mean[2])

    @staticmethod
    def importance_sample_batch(grid, x, nsamples, rng):
        """
        向量化重要性采样：按涡量幅值分布一次性采样 nsamples 个点。
        返回:
            y_batch: shape (nsamples, 3)  采样点坐标
            pdf_batch: shape (nsamples,)  对应的概率密度
        """
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        dx = grid.dx
        ox, oy, oz = grid.ox, grid.oy, grid.oz
        # 构建权重数组（涡量幅值）
        vort_mag = np.sqrt(np.sum(grid.vort**2, axis=-1))  # (nz, ny, nx)
        weights = vort_mag.flatten()
        total_weight = weights.sum()
        if total_weight < 1e-12:
            # 无涡量区域，回退到均匀采样
            return BiotSavart3D.uniform_sample_batch(grid, nsamples)
        pdf_weights = weights / total_weight
        # 一次性采样 nsamples 个网格索引（带权重，允许重复）
        indices = np.random.choice(nx * ny * nz, size=nsamples, p=pdf_weights)
        # 解码为 (k, j, i)
        k = indices // (nx * ny)
        j = (indices - k * nx * ny) // nx
        i = indices - k * nx * ny - j * nx
        # 在网格单元内均匀抖动
        xi = np.random.uniform(-0.5, 0.5, size=(nsamples, 3))
        y_batch = np.zeros((nsamples, 3))
        y_batch[:, 0] = ox + (i + 0.5 + xi[:, 0]) * dx
        y_batch[:, 1] = oy + (j + 0.5 + xi[:, 1]) * dx
        y_batch[:, 2] = oz + (k + 0.5 + xi[:, 2]) * dx
        # PDF = 选中该网格的概率 / 网格面积（连续密度）
        # 权重 pdf_weights[indices] 是离散概率，除以网格面积转换为连续密度
        cell_volume = dx ** 3
        pdf_batch = pdf_weights[indices] / cell_volume
        return y_batch, pdf_batch

    @staticmethod
    def uniform_sample_batch(grid, nsamples):
        """均匀采样（备用）"""
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        dx = grid.dx
        ox, oy, oz = grid.ox, grid.oy, grid.oz
        y_batch = np.random.uniform(
            [ox, oy, oz],
            [ox + nx*dx, oy + ny*dx, oz + nz*dx],
            size=(nsamples, 3)
        )
        volume = nx * ny * nz * dx ** 3
        pdf_batch = np.full(nsamples, 1.0 / volume)
        return y_batch, pdf_batch