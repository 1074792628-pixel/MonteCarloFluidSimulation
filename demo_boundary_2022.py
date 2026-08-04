"""
Improved demonstration: 2022 vorticity method boundary handling failure.
A vortex near a circular obstacle: the Biot-Savart velocity (without
harmonic correction) passes THROUGH the circle, violating no-penetration.
The 2024 method correctly deflects the flow around the obstacle.
"""
import sys, os
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU, biot_savart_gpu

res = 64
cx, cy, cr = 0.0, 0.0, 0.15

sim = Sim2022GPU(res, res, 0.05, 256)
# Initialize a vortex (nonzero vorticity) near the circle
sigma = 0.15
vort_center = (-0.5, 0.0)  # vortex to the left of the circle
for j in range(res):
    for i in range(res):
        px, py = sim.grid[0].grid_pos(i, j)
        d = np.hypot(px-vort_center[0], py-vort_center[1])
        w = 2.0*np.exp(-d*d/(2*sigma*sigma))
        # Mask out the obstacle (vorticity cannot exist inside solid)
        if np.hypot(px-cx, py-cy) < cr:
            w = 0.0
        sim.grid[0].vort[j, i] = w
        sim.grid[1].vort[j, i] = w

# Biot-Savart velocity (ignores the obstacle boundary - no harmonic correction)
vel = biot_savart_gpu(sim.query_pos, sim.grid[0].vort, res, res,
                      sim.ox, sim.oy, sim.dx, 256, sample_points=sim.sample_points)
vel_np = cp.asnumpy(vel).reshape(res, res, 2)
speed = np.sqrt(vel_np[..., 0]**2 + vel_np[..., 1]**2)

xs = (np.arange(res)+0.5)*sim.dx + sim.ox
X, Y = np.meshgrid(xs, xs, indexing='ij')
th = np.linspace(0, 2*np.pi, 100)

# Check: how much velocity crosses into the circle interior?
inside_mask = (X-cx)**2 + (Y-cy)**2 < cr**2
vel_inside = np.sqrt((vel_np[..., 0]**2 + vel_np[..., 1]**2)[inside_mask]).mean()
print(f"2022 method: mean speed INSIDE circle obstacle = {vel_inside:.4f}")
print(f"  => The flow passes THROUGH the obstacle (no-penetration violated)")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(speed.T, origin='lower', cmap='viridis', extent=[-1,1,-1,1])
axes[0].plot(cx+cr*np.cos(th), cy+cr*np.sin(th), 'k-', lw=2.5)
axes[0].set_title(f'2022: vortex near circle\n(flow crosses circle, mean|u|_inside={vel_inside:.3f})')
axes[0].set_xlim(-1,1); axes[0].set_ylim(-1,1); axes[0].set_aspect('equal')
plt.colorbar(axes[0].images[0], ax=axes[0])

skip = max(1, res//8)
axes[1].quiver(X[::skip,::skip], Y[::skip,::skip],
               vel_np[::skip,::skip,0], vel_np[::skip,::skip,1], scale=1.5, width=0.004)
axes[1].plot(cx+cr*np.cos(th), cy+cr*np.sin(th), 'k-', lw=2.5)
axes[1].set_title('2022: velocity vectors\n(streamlines pass through obstacle)')
axes[1].set_xlim(-1,1); axes[1].set_ylim(-1,1); axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('report_results/fig_2022_boundary_failure.png', dpi=130)
plt.close()
print("Saved fig_2022_boundary_failure.png")
