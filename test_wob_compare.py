"""
Compare: WoS projection vs WoB projection for MC fluid simulation.
Both use the same advection and diffusion steps.
"""
import numpy as np, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fluid_sim_2022_numba import FluidSim2022N
from fluid_sim_wob import FluidSimWoB
from scipy.ndimage import gaussian_filter as gs

res = 16
nu = 0.05
dt = 0.03
n_steps = 20

os.makedirs('results_wob', exist_ok=True)

# Taylor-Green initial condition
xs = np.linspace(-1, 1, res)
X, Y = np.meshgrid(xs, xs, indexing='ij')
vel0 = np.zeros((res, res, 2))
vel0[..., 0] = -np.cos(np.pi * X) * np.sin(np.pi * Y)
vel0[..., 1] = np.sin(np.pi * X) * np.cos(np.pi * Y)

# WoS-based simulation
sim_wos = FluidSim2022N(res, nu, dt, n_walks=512)
sim_wos.set_velocity(vel0.copy())

# WoB-based simulation
sim_wob = FluidSimWoB(res, nu, dt, n_particular=512, n_rays=128, smooth_sigma=0.3)
sim_wob.set_velocity(vel0.copy())

print("=" * 60)
print("WoS vs WoB Monte Carlo Fluids")
print("=" * 60)
print(f"{'Step':>5} | {'KE_WoS':>9} {'KE_WoB':>9} | {'Div_WoS':>9} {'Div_WoB':>9}")
print("-" * 55)

e_wos, e_wob = [], []
d_wos, d_wob = [], []

for step in range(n_steps):
    sim_wos.step()
    sim_wob.step()

    e_wos.append(sim_wos.kinetic_energy())
    e_wob.append(sim_wob.kinetic_energy())
    d_wos.append(sim_wos.divergence_error())
    d_wob.append(sim_wob.divergence_error())

    if step % 5 == 0 or step == n_steps - 1:
        print(f"{step:5d} | {e_wos[-1]:9.2f} {e_wob[-1]:9.2f} | "
              f"{d_wos[-1]:9.6f} {d_wob[-1]:9.6f}")

# Final comparison figure
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

curl_wos = sim_wos.vorticity()
curl_wob = sim_wob.vorticity()
vmax = max(abs(curl_wos).max(), abs(curl_wob).max(), 1e-6)

for row, (sim, name, curl) in enumerate([
    (sim_wos, 'WoS Projection', curl_wos),
    (sim_wob, 'WoB Projection', curl_wob)]):
    axes[row,0].imshow(curl.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
    axes[row,0].set_title(f'{name}\nVorticity')

    speed = np.sqrt(sim.velocity[...,0]**2 + sim.velocity[...,1]**2)
    axes[row,1].imshow(speed.T, origin='lower', cmap='viridis',
                       extent=[-1,1,-1,1])
    axes[row,1].set_title('Speed |u|')

    div = sim.divergence_error()
    axes[row,2].text(0.5, 0.5, f'mean|div| = {div:.6f}',
                     ha='center', va='center', fontsize=14,
                     transform=axes[row,2].transAxes)
    axes[row,2].set_title('Divergence')

    skip = max(1, res//8)
    axes[row,3].quiver(X[::skip,::skip], Y[::skip,::skip],
                       sim.velocity[::skip,::skip,0],
                       sim.velocity[::skip,::skip,1],
                       scale=2.0, width=0.005)
    axes[row,3].set_title('Vector field')
    axes[row,3].set_aspect('equal')

plt.tight_layout()
plt.savefig('results_wob/wos_vs_wob_final.png', dpi=120)

# Energy & divergence plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(e_wos, 'b.-', label='WoS (Pressure Poisson)')
axes[0].plot(e_wob, 'r.-', label='WoB (Walk-on-Boundary)')
axes[0].set_xlabel('Step'); axes[0].set_ylabel('KE')
axes[0].legend(); axes[0].set_title('Kinetic Energy')
axes[0].set_yscale('log')

axes[1].plot(d_wos, 'b.-', label='WoS')
axes[1].plot(d_wob, 'r.-', label='WoB')
axes[1].set_xlabel('Step'); axes[1].set_ylabel('mean |div u|')
axes[1].legend(); axes[1].set_title('Divergence Error')
axes[1].set_yscale('log')

diff = abs(curl_wos) - abs(curl_wob)
dmax = max(abs(diff).max(), 1e-6)
im = axes[2].imshow(diff.T, origin='lower', cmap='RdBu_r',
                     vmin=-dmax, vmax=dmax, extent=[-1,1,-1,1])
axes[2].set_title('|ω_WoS| - |ω_WoB|')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig('results_wob/wos_vs_wob_energy.png', dpi=120)

print(f"\n=== Summary ===")
print(f"WoS: Final KE={e_wos[-1]:.2f}, Mean|div|={np.mean(d_wos):.6f}, "
      f"Time={sim_wos.stats['total_time']:.1f}s")
print(f"WoB: Final KE={e_wob[-1]:.2f}, Mean|div|={np.mean(d_wob):.6f}, "
      f"Time={sim_wob.stats['total_time']:.1f}s")
print(f"\nResults saved to 'results_wob/'")
