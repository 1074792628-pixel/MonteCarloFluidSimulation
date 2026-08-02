"""
Consolidated quantitative error analysis for 2022 (GPU) vs 2024 (WoB).
Viscous Taylor-Green (nu=0.05), analytical reference.
Plots: (1) error vs samples, (2) error vs resolution, (3) time-step error evolution.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU
from fluid_sim_wob import FluidSimWoB

def tg_w(X, Y, t, nu):
    return 2*np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y)*np.exp(-2*np.pi**2*nu*t)

def curl2d(u, v, h):
    w = np.zeros_like(u)
    w[1:-1, 1:-1] = (v[2:,1:-1]-v[:-2,1:-1])/(2*h) - (u[1:-1,2:]-u[1:-1,:-2])/(2*h)
    return w

def err_2022(res, nmc, steps, nu=0.05):
    sim = Sim2022GPU(res, res, 0.05, nmc, nu=nu)
    sim._init_taylor_green()
    xs = (np.arange(res)+0.5)*sim.dx + sim.ox
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    errs = []
    for s in range(steps):
        sim.step()
        errs.append(np.sqrt(np.mean((sim.vorticity - tg_w(XX,YY,(s+1)*0.05,nu))**2)))
    return errs

def err_2024(res, n_part, n_rays, steps, nu=0.05):
    xs = np.linspace(-1,1,res)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    vel0 = np.zeros((res,res,2))
    vel0[...,0] = -np.cos(np.pi*X)*np.sin(np.pi*Y)
    vel0[...,1] = np.sin(np.pi*X)*np.cos(np.pi*Y)
    sim = FluidSimWoB(res, nu, 0.05, n_particular=n_part, n_rays=n_rays, smooth_sigma=0.3)
    sim.set_velocity(vel0.copy())
    h = sim.h
    errs = []
    for s in range(steps):
        sim.step()
        w = curl2d(sim.velocity[...,0], sim.velocity[...,1], h)
        errs.append(np.sqrt(np.mean((w - tg_w(X,Y,(s+1)*0.05,nu))**2)))
    return errs

res = 64
steps = 20

# Error vs samples (res=64)
nmc_list = [16, 64, 256, 1024]
e22_samples = [err_2022(res, n, steps)[-1] for n in nmc_list]
samp24 = [(32,16),(128,64),(512,128),(2048,256)]
e24_samples = []
for p, r in samp24:
    e = err_2024(res, p, r, steps)
    e24_samples.append(e[-1])

# Error vs resolution (fixed samples)
res_list = [16, 32, 64]
e22_res = [err_2022(r, 256, steps)[-1] for r in res_list]
e24_res = []
for r in res_list:
    e = err_2024(r, 512, 128, steps)
    e24_res.append(e[-1])

# Time evolution at res=64, moderate samples
t22 = err_2022(64, 256, 20)
t24 = err_2024(64, 512, 128, 20)

# Save data
np.savez('report_results/error_analysis.npz',
         nmc_list=np.array(nmc_list), e22_samples=np.array(e22_samples),
         e24_samples=np.array(e24_samples),
         res_list=np.array(res_list), e22_res=np.array(e22_res),
         e24_res=np.array(e24_res),
         t22=np.array(t22), t24=np.array(t24))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].loglog(nmc_list, e22_samples, 'b-o', label='2022 (nmc)', linewidth=2, markersize=6)
axes[0].semilogy([s[0] for s in samp24], e24_samples, 'r-s', label='2024 (n_part)', linewidth=2, markersize=6)
axes[0].set_xlabel('MC samples'); axes[0].set_ylabel('Vorticity RMSE (t=1.0)')
axes[0].legend(); axes[0].set_title('Error vs MC samples (res=64)')
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(res_list, e22_res, 'b-o', label='2022', linewidth=2, markersize=6)
axes[1].semilogy(res_list, e24_res, 'r-s', label='2024', linewidth=2, markersize=6)
axes[1].set_xlabel('Grid resolution'); axes[1].set_ylabel('Vorticity RMSE (t=1.0)')
axes[1].legend(); axes[1].set_title('Error vs resolution')
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(t22, 'b-o', label='2022', linewidth=2, markersize=4)
axes[2].semilogy(t24, 'r-s', label='2024', linewidth=2, markersize=4)
axes[2].set_xlabel('Time step'); axes[2].set_ylabel('Vorticity RMSE')
axes[2].legend(); axes[2].set_title('Error evolution (res=64)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_results/error_analysis.png', dpi=150)
plt.close()

print("=== Error analysis summary (res=64, nu=0.05, t=1.0) ===")
print(f"2022 vs samples: nmc {nmc_list[0]}-{nmc_list[-1]} -> RMSE {e22_samples[0]:.3f}-{e22_samples[-1]:.3f}")
print(f"2024 vs samples: part {samp24[0][0]}-{samp24[-1][0]} -> RMSE {e24_samples[0]:.3f}-{e24_samples[-1]:.3f}")
print(f"2022 final RMSE (nmc=256, res=64): {e22_samples[2]:.4f}")
print(f"2024 final RMSE (512p+128r, res=64): {e24_samples[2]:.4f}")
print(f"Saved report_results/error_analysis.png")
