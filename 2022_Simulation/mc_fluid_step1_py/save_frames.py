import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.types import Vec2
from core.grid import Grid
from core.biot_savart import BiotSavart
import random as pyrandom

class RNG:
    def __init__(self): self.uni = pyrandom.uniform
    def uniform_in_box(self, xmin, xmax, ymin, ymax):
        return Vec2(self.uni(xmin,xmax), self.uni(ymin,ymax))

class Sim2022:
    def __init__(self, nx, ny, dt, nmc):
        self.nx, self.ny, self.dt, self.nmc = nx, ny, dt, nmc
        self.dx = 2.0 / nx
        self.grid = [Grid(nx, ny, self.dx, -1.0, -1.0) for _ in range(2)]
        self.cur = 0
        self.rng = RNG()
        c1 = Vec2(-0.7, 1.0/6.0)
        c2 = Vec2(-0.7, -1.0/6.0)
        radius = 0.8 / 6.0
        for j in range(ny):
            for i in range(nx):
                p = self.grid[0].grid_pos(i, j)
                w = 0.0
                if (p - c1).norm() <= radius: w += 1.0
                if (p - c2).norm() <= radius: w -= 1.0
                self.grid[0].set_vort(i, j, w)
                self.grid[1].set_vort(i, j, w)
        self.frames = [self.vorticity.copy()]
    def step(self):
        prev, nxt = self.grid[self.cur], self.grid[1-self.cur]
        for j in range(self.ny):
            for i in range(self.nx):
                x = prev.grid_pos(i, j)
                v = BiotSavart.estimate_velocity(prev, x, self.nmc, self.rng)
                xb = x - v*self.dt
                w = prev.get_vort(xb)
                nxt.set_vort(i, j, w)
        self.cur = 1 - self.cur
        self.frames.append(self.vorticity.copy())
    @property
    def vorticity(self): return self.grid[self.cur].vort

res, dt, nmc, steps = 64, 0.05, 128, 40
sim = Sim2022(res, res, dt, nmc)
for _ in range(steps):
    sim.step()

os.makedirs('compare_output/frames', exist_ok=True)
vmax = 1.0
for fstep in [0, 10, 20, 30, 39]:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sim.frames[fstep].T, origin='lower', cmap='RdBu_r',
              vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
    ax.set_title(f'2022 t={fstep*dt:.1f}')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'compare_output/frames/frame_{fstep:03d}.png', dpi=100)
    plt.close()
print("Saved 2022 frames")
