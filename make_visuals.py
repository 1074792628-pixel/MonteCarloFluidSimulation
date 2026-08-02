"""
Generate comprehensive visualization figures for the 2022 vs 2024 comparison.
Includes: vorticity field snapshots, error evolution, KE decay, error vs samples.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

# ============ Helpers ============
def read_vel(path):
    with open(path, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 2)

def curl(u, v, h, res):
    w = np.zeros((res, res))
    w[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2*h) - (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*h)
    return w

# ============ Run 2022 & collect fields ============
res = 64
sim = Sim2022GPU(res, res, 0.05, 256, nu=0.05)
sim._init_taylor_green()
xs2 = (np.arange(res)+0.5)*sim.dx + sim.ox
X2, Y2 = np.meshgrid(xs2, xs2, indexing='ij')
w0_22 = 2*np.pi*np.cos(np.pi*X2)*np.cos(np.pi*Y2)

w22_fields = [sim.vorticity.copy()]
for s in range(20):
    sim.step()
    w22_fields.append(sim.vorticity.copy())

# 2024 fields from saved runs
L = 2.4; h = L/res
xs = np.linspace(-L/2+h/2, L/2-h/2, res)
X, Y = np.meshgrid(xs, xs, indexing='xy')
k = 2*np.pi/L
w24_fields = []
for s in range(21):
    vel = read_vel(f'VelMCFluids/results/results_tg_compare/raw/velocity_{s}.vector')
    u = vel[:, 0].reshape(res, res); v = vel[:, 1].reshape(res, res)
    w24_fields.append(curl(u, v, h, res))

# ============ Figure 1: Field snapshots ============
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
times = [0, 5, 10, 20]  # t = 0, 0.25, 0.5, 1.0
for j, s in enumerate(times):
    t = s*0.05
    # Analytical 2022
    wa = w0_22*np.exp(-2*np.pi**2*0.05*t)
    # Analytical 2024
    wb = 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*t)
    vmax = max(np.abs(w22_fields[s]).max(), np.abs(w24_fields[s]).max(),
               np.abs(wa).max(), np.abs(wb).max())
    # Row 0: analytical
    axes[0, j].imshow(wa.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
    axes[0, j].set_title(f't={t:.2f}  (analytical)')
    axes[0, j].axis('off')
    # Row 1: 2022
    axes[1, j].imshow(w22_fields[s].T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
    axes[1, j].set_title(f'2022 method')
    axes[1, j].axis('off')
    # Row 2: 2024
    axes[2, j].imshow(w24_fields[s].T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[-1.2,1.2,-1.2,1.2])
    axes[2, j].set_title(f'2024 method')
    axes[2, j].axis('off')
axes[0,0].set_ylabel('Analytical\n(reference)', fontsize=11)
axes[1,0].set_ylabel('2022\nmethod', fontsize=11)
axes[2,0].set_ylabel('2024\nmethod', fontsize=11)
plt.tight_layout()
plt.savefig('report_results/fig_field_comparison.png', dpi=130)
plt.close()

# ============ Figure 2: Error evolution + KE ============
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
t = (np.arange(20)+1)*0.05
e22 = np.array([np.sqrt(np.mean((w22_fields[s+1] - w0_22*np.exp(-2*np.pi**2*0.05*(s+1)*0.05))**2)) for s in range(20)])
e24 = np.array([np.sqrt(np.mean((w24_fields[s+1] - 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*(s+1)*0.05))**2)) for s in range(20)])

axes[0].semilogy(t, e22, 'b-o', label='2022 method', linewidth=2)
axes[0].semilogy(t, e24, 'r-s', label='2024 method', linewidth=2)
axes[0].set_xlabel('Time t'); axes[0].set_ylabel('Vorticity RMSE vs analytical')
axes[0].legend(); axes[0].set_title('Error evolution')
axes[0].grid(True, alpha=0.3)

# KE decay vs analytical
ke22 = np.array([0.5*np.sum(w22_fields[s]**2) for s in range(21)])
ke22_ana = np.array([0.5*np.sum((w0_22*np.exp(-2*np.pi**2*0.05*s*0.05))**2) for s in range(21)])
ke24 = np.array([0.5*np.sum(w24_fields[s]**2) for s in range(21)])
ke24_ana = np.array([0.5*np.sum((2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*s*0.05))**2) for s in range(21)])
axes[1].plot(t, ke22[1:]/ke22[0], 'b-o', label='2022 method', linewidth=2)
axes[1].plot(t, ke24[1:]/ke24[0], 'r-s', label='2024 method', linewidth=2)
axes[1].plot(t, ke22_ana[1:]/ke22_ana[0], 'b--', label='2022 analytical', linewidth=1.5)
axes[1].plot(t, ke24_ana[1:]/ke24_ana[0], 'r--', label='2024 analytical', linewidth=1.5)
axes[1].set_xlabel('Time t'); axes[1].set_ylabel('KE / KE(0)')
axes[1].legend(); axes[1].set_title('KE decay (viscous)')
axes[1].grid(True, alpha=0.3)

# Divergence error (2024)
div24 = []
for s in range(1, 21):
    vel = read_vel(f'VelMCFluids/results/results_tg_compare/raw/velocity_{s}.vector')
    u = vel[:, 0].reshape(res, res); v = vel[:, 1].reshape(res, res)
    d = (u[1:-1, 2:] - u[1:-1, :-2])/(2*h) + (v[2:, 1:-1] - v[:-2, 1:-1])/(2*h)
    div24.append(np.mean(np.abs(d)))
axes[2].plot(t, div24, 'r-s', label='2024 method div error', linewidth=2)
axes[2].axhline(0, color='k', ls='--', lw=1)
axes[2].set_xlabel('Time t'); axes[2].set_ylabel('Mean |div u|')
axes[2].legend(); axes[2].set_title('2024 divergence error')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_results/fig_error_ke.png', dpi=130)
plt.close()

print("Saved fig_field_comparison.png, fig_error_ke.png")
