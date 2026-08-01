from .grid import Grid
from .biot_savart import BiotSavart
from .types import Vec2, RNG
import numpy as np

class Simulator:
    def __init__(self, nx, ny, dt):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = 2.0 / nx                     # domain [-1,1]^2
        ox, oy = -1.0, -1.0
        self.grid = [Grid(nx, ny, self.dx, ox, oy) for _ in range(2)]
        self.cur = 0
        self.nmc = 256
        self.paused = False
        self.rng = RNG()
        self._init_vorticity()

    def _init_vorticity(self):
        sigma = 0.2
        centers = [Vec2(-0.5, 0.0), Vec2(0.5, 0.0)]
        for j in range(self.ny):
            for i in range(self.nx):
                p = self.grid[0].grid_pos(i,j)
                w = 0.0
                for c in centers:
                    d = p - c
                    w += np.exp(-d.norm2() / (2*sigma*sigma))
                self.grid[0].set_vort(i,j,w)
                self.grid[1].set_vort(i,j,w)

    def step(self):
        prev = self.grid[self.cur]
        nxt = self.grid[1-self.cur]
        for j in range(self.ny):
            for i in range(self.nx):
                x = prev.grid_pos(i,j)
                v = BiotSavart.estimate_velocity(prev, x, self.nmc, self.rng)
                xb = x - v * self.dt
                w = prev.get_vort(xb)
                nxt.set_vort(i,j,w)
        self.cur = 1 - self.cur

    @property
    def vorticity(self):
        return self.grid[self.cur].vort

    def reset(self):
        self.cur = 0
        self._init_vorticity()