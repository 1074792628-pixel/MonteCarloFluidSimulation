import numpy as np
from .types import Vec2, RNG
from .grid import Grid

class BiotSavart:
    @staticmethod
    def kernel(x: Vec2, y: Vec2) -> Vec2:
        r = y - x
        r2 = r.norm2() + 1e-12  # avoid zero division
        # 2D kernel: (r.y, -r.x) / (2 pi r^2)
        return Vec2(r.y / (2*np.pi*r2), -r.x / (2*np.pi*r2))

    @staticmethod
    def estimate_velocity(prev_vort: Grid, x: Vec2, nsamples: int, rng: RNG) -> Vec2:
        area = prev_vort.area
        pdf = 1.0 / area
        sum_v = Vec2()
        for _ in range(nsamples):
            y = rng.uniform_in_box(
                prev_vort.ox, prev_vort.ox + prev_vort.nx*prev_vort.dx,
                prev_vort.oy, prev_vort.oy + prev_vort.ny*prev_vort.dx)
            w = prev_vort.get_vort(y)
            k = BiotSavart.kernel(x, y)
            sum_v = sum_v + k * (w / pdf)
        return sum_v * (1.0 / nsamples)