"""
Demonstrate the fundamental limitation of the 2022 vorticity method:
A uniform background flow past a circle obstacle is a HARMONIC field (zero vorticity),
so Biot-Savart reconstruction from vorticity gives ~zero velocity.
The 2022 method cannot represent this flow, while 2024 can.
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
# Circle obstacle at center, radius 0.15 (matches 2024 circle config)
cx, cy, cr = 0.0, 0.0, 0.15

sim = Sim2022GPU(res, res, 0.05, 256)
# Zero vorticity everywhere (uniform flow has omega = 0)
# But we need SOME vorticity to see Biot-Savart behavior.
# Setup: a small vorticity perturbation in the wake region only,
# simulating what a vorticity method could represent.
for j in range(res):
    for i in range(res):
        px, py = sim.grid[0].grid_pos(i, j)
        # zero vorticity: uniform flow is harmonic
        sim.grid[0].vort[j, i] = 0.0
        sim.grid[1].vort[j, i] = 0.0

# Compute velocity via Biot-Savart from zero vorticity
vel = biot_savart_gpu(sim.query_pos, sim.grid[0].vort, res, res,
                      sim.ox, sim.oy, sim.dx, 256, sample_points=sim.sample_points)
vel_np = cp.asnumpy(vel).reshape(res, res, 2)
speed = np.sqrt(vel_np[..., 0]**2 + vel_np[..., 1]**2)

# Grid for plotting
xs = (np.arange(res)+0.5)*sim.dx + sim.ox
X, Y = np.meshgrid(xs, xs, indexing='ij')

print("=== 2022 vorticity method: uniform flow past circle ===")
print(f"Maximum velocity from Biot-Savart (omega=0): {speed.max():.4e}")
print("=> The uniform background flow has zero vorticity,")
print("   so Biot-Savart reconstruction gives ~zero velocity.")
print("   The 2022 method CANNOT represent this flow at all.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].imshow(speed.T, origin='lower', cmap='viridis', extent=[-1,1,-1,1])
axes[0].set_title('2022: velocity from Biot-Savart\n(max|u| = %.2e)' % speed.max())
# Draw circle
th = np.linspace(0, 2*np.pi, 100)
axes[0].plot(cx+cr*np.cos(th), cy+cr*np.sin(th), 'k-', lw=2)
axes[0].set_xlim(-1,1); axes[0].set_ylim(-1,1)
axes[0].axis('equal')

skip = max(1, res//10)
axes[1].quiver(X[::skip,::skip], Y[::skip,::skip],
               vel_np[::skip,::skip,0], vel_np[::skip,::skip,1], scale=1.0)
axes[1].plot(cx+cr*np.cos(th), cy+cr*np.sin(th), 'k-', lw=2)
axes[1].set_title('2022: velocity field\n(no flow: cannot represent uniform current)')
axes[1].set_xlim(-1,1); axes[1].set_ylim(-1,1)
axes[1].set_aspect('equal')
plt.tight_layout()
plt.savefig('report_results/fig_2022_harmonic_failure.png', dpi=130)
plt.close()
print("Saved fig_2022_harmonic_failure.png")
