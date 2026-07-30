import numpy as np
from .types import Vec2

class Grid:
    def __init__(self, nx, ny, dx, ox, oy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.ox = ox
        self.oy = oy
        self.vort = np.zeros((ny, nx), dtype=np.float64)  # row-major

    def get_vort(self, pos: Vec2) -> float:
        fx = (pos.x - self.ox) / self.dx - 0.5
        fy = (pos.y - self.oy) / self.dx - 0.5
        i0 = int(np.floor(fx))
        j0 = int(np.floor(fy))
        tx = fx - i0
        ty = fy - j0
        i0 = max(0, min(i0, self.nx-2))
        j0 = max(0, min(j0, self.ny-2))
        i1, j1 = i0+1, j0+1
        v00 = self.vort[j0, i0]
        v10 = self.vort[j0, i1]
        v01 = self.vort[j1, i0]
        v11 = self.vort[j1, i1]
        return (1-ty)*((1-tx)*v00 + tx*v10) + ty*((1-tx)*v01 + tx*v11)

    def set_vort(self, i, j, val):
        self.vort[j, i] = val

    def grid_pos(self, i, j) -> Vec2:
        return Vec2(self.ox + (i+0.5)*self.dx, self.oy + (j+0.5)*self.dx)

    @property
    def area(self):
        return (self.nx * self.dx) * (self.ny * self.dx)