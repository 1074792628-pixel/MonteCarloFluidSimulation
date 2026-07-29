"""
Generate comparison figures for the report.
WoS (pressure Poisson) vs WoB (Walk-on-Boundary) projection.
"""
import numpy as np, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fluid_sim_2022_numba import FluidSim2022N
from fluid_sim_wob import FluidSimWoB
from scipy.ndimage import gaussian_filter as gs

os.makedirs('report_results', exist_ok=True)

# ─── Test 1: Poisson solve accuracy ────────────────────────────────

res = 16
h = 2.0 / (res - 1)
xs = np.linspace(-1, 1, res)
X, Y = np.meshgrid(xs, xs, indexing='ij')

f_exact = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
p_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

from numba_wos import wos_poisson_grid_numba, wob_poisson_grid_numba

p_wos = gs(wos_poisson_grid_numba(X, Y, f_exact, res, n_walks=512, eps=0.02),
           sigma=0.4, mode='reflect')
p_wob = gs(wob_poisson_grid_numba(X, Y, f_exact, res, n_particular=512, n_rays=128),
           sigma=0.4, mode='reflect')

err_wos = np.abs(p_wos - p_exact)
err_wob = np.abs(p_wob - p_exact)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
vmax = 1.0

axes[0,0].imshow(p_exact.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
axes[0,0].set_title('Exact: sin(πx)sin(πy)')
axes[0,1].imshow(p_wos.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
axes[0,1].set_title(f'WoS (RMSE={np.sqrt(np.mean(err_wos**2)):.3f})')
axes[0,2].imshow(p_wob.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
axes[0,2].set_title(f'WoB (RMSE={np.sqrt(np.mean(err_wob**2)):.3f})')

axes[1,0].axis('off')
im1 = axes[1,1].imshow(err_wos.T, origin='lower', cmap='hot', extent=[-1,1,-1,1])
axes[1,1].set_title('WoS |error|')
plt.colorbar(im1, ax=axes[1,1])
im2 = axes[1,2].imshow(err_wob.T, origin='lower', cmap='hot', extent=[-1,1,-1,1])
axes[1,2].set_title('WoB |error|')
plt.colorbar(im2, ax=axes[1,2])

plt.tight_layout()
plt.savefig('report_results/poisson_accuracy.png', dpi=120)
plt.close()

# ─── Test 2: Single-step projection ────────────────────────────────

# Divergent velocity: u = [x, 0], ∇·u = 1
u_test = np.zeros((res, res, 2))
u_test[..., 0] = X

src = np.ones((res, res))  # ∇·u = 1

# WoS Poisson solve + gradient
p_wos = wos_poisson_grid_numba(X, Y, src, res, n_walks=512, eps=0.02)
p_wos = gs(p_wos, sigma=0.4, mode='reflect')
px_wos, py_wos = np.gradient(p_wos, h, h)
u_wos_proj = np.zeros_like(u_test)
u_wos_proj[..., 0] = u_test[..., 0] - px_wos
u_wos_proj[..., 1] = u_test[..., 1] - py_wos

# WoB Poisson solve + gradient
p_wob_s = wob_poisson_grid_numba(X, Y, src, res, n_particular=512, n_rays=128)
p_wob_s = gs(p_wob_s, sigma=0.4, mode='reflect')
px_wob, py_wob = np.gradient(p_wob_s, h, h)
u_wob_proj = np.zeros_like(u_test)
u_wob_proj[..., 0] = u_test[..., 0] - px_wob
u_wob_proj[..., 1] = u_test[..., 1] - py_wob

from numba_wos import divergence_2d_numba
div_before = np.mean(np.abs(divergence_2d_numba(u_test[...,0], u_test[...,1], h)[1:-1,1:-1]))
div_wos = np.mean(np.abs(divergence_2d_numba(u_wos_proj[...,0], u_wos_proj[...,1], h)[1:-1,1:-1]))
div_wob = np.mean(np.abs(divergence_2d_numba(u_wob_proj[...,0], u_wob_proj[...,1], h)[1:-1,1:-1]))

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
vmax_d = 2.0
axes[0,0].imshow(divergence_2d_numba(u_test[...,0], u_test[...,1], h).T,
                  origin='lower', cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, extent=[-1,1,-1,1])
axes[0,0].set_title(f'Input ∇·u (mean|div|={div_before:.3f})')
axes[0,1].imshow(divergence_2d_numba(u_wos_proj[...,0], u_wos_proj[...,1], h).T,
                  origin='lower', cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, extent=[-1,1,-1,1])
axes[0,1].set_title(f'WoS projected ∇·u (mean|div|={div_wos:.3f})')
axes[0,2].imshow(divergence_2d_numba(u_wob_proj[...,0], u_wob_proj[...,1], h).T,
                  origin='lower', cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, extent=[-1,1,-1,1])
axes[0,2].set_title(f'WoB projected ∇·u (mean|div|={div_wob:.3f})')

skip = max(1, res//6)
axes[1,0].quiver(X[::skip,::skip], Y[::skip,::skip],
                  u_test[::skip,::skip,0], u_test[::skip,::skip,1],
                  scale=1.5, width=0.008)
axes[1,0].set_title('Input u = [x, 0]')
axes[1,1].quiver(X[::skip,::skip], Y[::skip,::skip],
                  u_wos_proj[::skip,::skip,0], u_wos_proj[::skip,::skip,1],
                  scale=1.5, width=0.008)
axes[1,1].set_title('WoS projected u')
axes[1,2].quiver(X[::skip,::skip], Y[::skip,::skip],
                  u_wob_proj[::skip,::skip,0], u_wob_proj[::skip,::skip,1],
                  scale=1.5, width=0.008)
axes[1,2].set_title('WoB projected u')
for ax in axes[1,:]:
    ax.set_aspect('equal')
    ax.set_xlim(-1,1); ax.set_ylim(-1,1)

plt.tight_layout()
plt.savefig('report_results/single_step_projection.png', dpi=120)
plt.close()

print(f"Single-step projection:")
print(f"  Input:     mean|div| = {div_before:.4f}")
print(f"  WoS proj:  mean|div| = {div_wos:.4f}")
print(f"  WoB proj:  mean|div| = {div_wob:.4f}")

# ─── Test 3: Full simulation comparison ─────────────────────────────

res_sim = 12
nu = 0.05
dt = 0.03
n_steps = 15

xs = np.linspace(-1, 1, res_sim)
X, Y = np.meshgrid(xs, xs, indexing='ij')
vel0 = np.zeros((res_sim, res_sim, 2))
vel0[..., 0] = -np.cos(np.pi * X) * np.sin(np.pi * Y)
vel0[..., 1] = np.sin(np.pi * X) * np.cos(np.pi * Y)

sim_wos = FluidSim2022N(res_sim, nu, dt, n_walks=512)
sim_wos.set_velocity(vel0.copy())

sim_wob = FluidSimWoB(res_sim, nu, dt, n_particular=512, n_rays=128, smooth_sigma=0.3)
sim_wob.set_velocity(vel0.copy())

e_wos, e_wob = [], []
d_wos, d_wob = [], []

for step in range(n_steps):
    sim_wos.step()
    sim_wob.step()
    e_wos.append(sim_wos.kinetic_energy())
    e_wob.append(sim_wob.kinetic_energy())
    d_wos.append(sim_wos.divergence_error())
    d_wob.append(sim_wob.divergence_error())
    if step % 5 == 0:
        print(f"Step {step}: WoS KE={e_wos[-1]:.1f} div={d_wos[-1]:.4f}  |  WoB KE={e_wob[-1]:.1f} div={d_wob[-1]:.4f}")

# Final comparison figure
from numba_wos import curl_2d_numba
curl_wos = curl_2d_numba(sim_wos.velocity[...,0], sim_wos.velocity[...,1], sim_wos.h)
curl_wob = curl_2d_numba(sim_wob.velocity[...,0], sim_wob.velocity[...,1], sim_wob.h)
vmax_c = max(abs(curl_wos).max(), abs(curl_wob).max(), 1e-6)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for row, (sim, name, curl) in enumerate([
    (sim_wos, 'WoS Pressure Poisson', curl_wos),
    (sim_wob, 'WoB Walk-on-Boundary', curl_wob)]):
    axes[row,0].imshow(curl.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax_c, vmax=vmax_c, extent=[-1,1,-1,1])
    axes[row,0].set_title(f'{name}\nVorticity ω')
    speed = np.sqrt(sim.velocity[...,0]**2 + sim.velocity[...,1]**2)
    axes[row,1].imshow(speed.T, origin='lower', cmap='viridis', extent=[-1,1,-1,1])
    axes[row,1].set_title('Speed |u|')
    skip = max(1, res_sim//6)
    axes[row,2].quiver(X[::skip,::skip], Y[::skip,::skip],
                       sim.velocity[::skip,::skip,0], sim.velocity[::skip,::skip,1],
                       scale=2.0, width=0.005)
    axes[row,2].set_title('Vector field')
    axes[row,2].set_aspect('equal')
    axes[row,2].set_xlim(-1,1); axes[row,2].set_ylim(-1,1)

# Energy & divergence
axes[0,3].plot(e_wos, 'b.-', label='WoS')
axes[0,3].plot(e_wob, 'r.-', label='WoB')
axes[0,3].set_xlabel('Step'); axes[0,3].set_ylabel('Kinetic Energy')
axes[0,3].legend(); axes[0,3].set_title('Kinetic Energy')
axes[0,3].set_yscale('log')

axes[1,3].plot(d_wos, 'b.-', label='WoS')
axes[1,3].plot(d_wob, 'r.-', label='WoB')
axes[1,3].set_xlabel('Step'); axes[1,3].set_ylabel('mean |div u|')
axes[1,3].legend(); axes[1,3].set_title('Divergence Error')
axes[1,3].set_yscale('log')

plt.tight_layout()
plt.savefig('report_results/wos_vs_wob_full.png', dpi=120)
plt.close()

# Energy + divergence separate figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(e_wos, 'b.-', label='WoS (Pressure Poisson)', linewidth=1.5)
axes[0].plot(e_wob, 'r.-', label='WoB (Walk-on-Boundary)', linewidth=1.5)
axes[0].set_xlabel('Time step'); axes[0].set_ylabel('Kinetic Energy')
axes[0].legend(); axes[0].set_title('Kinetic Energy Evolution')
axes[0].grid(True, alpha=0.3)
axes[1].plot(d_wos, 'b.-', label='WoS', linewidth=1.5)
axes[1].plot(d_wob, 'r.-', label='WoB', linewidth=1.5)
axes[1].set_xlabel('Time step'); axes[1].set_ylabel('Mean |∇·u|')
axes[1].legend(); axes[1].set_title('Divergence Error Evolution')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_results/energy_divergence_comparison.png', dpi=120)
plt.close()

print(f"\n=== Summary ===")
print(f"WoS: Final KE={e_wos[-1]:.1f}, Mean|div|={np.mean(d_wos):.4f}, Time={sim_wos.stats['total_time']:.1f}s")
print(f"WoB: Final KE={e_wob[-1]:.1f}, Mean|div|={np.mean(d_wob):.4f}, Time={sim_wob.stats['total_time']:.1f}s")
print("Results saved to 'report_results/'")
