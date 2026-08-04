"""
Comprehensive quantitative comparison: 2022 vs 2024 on viscous Taylor-Green.
Generates an extended metrics table + convergence figure.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

def read_vel(path):
    with open(path, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 2)
def curl(u, v, h, res):
    w = np.zeros((res, res))
    w[1:-1, 1:-1] = (v[1:-1,2:]-v[1:-1,:-2])/(2*h) - (u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    return w

# --- 2022 metrics ---
def m22(res, nmc, steps=20, nu=0.05):
    sim = Sim2022GPU(res, res, 0.05, nmc, nu=nu)
    sim._init_taylor_green()
    xs = (np.arange(res)+0.5)*sim.dx + sim.ox
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    h = sim.dx
    w0 = 2*np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y)
    for s in range(steps): sim.step()
    w = sim.vorticity; t = steps*0.05
    wa = w0*np.exp(-2*np.pi**2*nu*t)
    rmse = np.sqrt(np.mean((w-wa)**2))
    rel = rmse/abs(w0).max()
    ke_e = abs(0.5*np.sum(w**2)*h*h - 0.5*np.sum(wa**2)*h*h)/(0.5*np.sum(wa**2)*h*h)
    abs_circ_e = abs(np.sum(abs(w))*h*h - np.sum(abs(wa))*h*h)/(np.sum(abs(wa))*h*h)
    maxw_e = abs(w.max()-wa.max())/abs(wa.max())
    return rmse, rel, ke_e, abs_circ_e, maxw_e

# --- 2024 metrics ---
def m24(samples, res=64, steps=20, nu=0.05):
    L = 2.4; h = L/res
    xs = np.linspace(-L/2+h/2, L/2-h/2, res)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    k = 2*np.pi/L; t = steps*0.05
    wa = 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*nu*t)
    dn = {256:'results_tg_compare', 512:'results_tg_s512', 1024:'results_tg_s1024'}[samples]
    vel = read_vel(f'VelMCFluids/results/{dn}/raw/velocity_{steps}.vector')
    u = vel[:,0].reshape(res,res); v = vel[:,1].reshape(res,res)
    w = curl(u, v, h, res)
    rmse = np.sqrt(np.mean((w-wa)**2))
    rel = rmse/(2*k)
    ke_e = abs(0.5*np.sum(u**2+v**2)*h*h - 0.5*np.sum(wa**2)*h*h)/(0.5*np.sum(wa**2)*h*h)
    abs_circ_e = abs(np.sum(abs(w))*h*h - np.sum(abs(wa))*h*h)/(np.sum(abs(wa))*h*h)
    maxw_e = abs(w.max()-wa.max())/abs(wa.max())
    # divergence error
    div = np.abs((u[1:-1,2:]-u[1:-1,:-2])/(2*h) + (v[2:,1:-1]-v[:-2,1:-1])/(2*h)).mean()
    return rmse, rel, ke_e, abs_circ_e, maxw_e, div

# Compute
r22 = m22(64, 256)
r24_256 = m24(256)
r24_1024 = m24(1024)

print("=== Comprehensive metrics (res=64, t=1.0, nu=0.05) ===")
print(f"指标                | 2022(nmc=256) | 2024(256)  | 2024(1024)")
print(f"涡量RMSE            | {r22[0]:.4f}      | {r24_256[0]:.4f}     | {r24_1024[0]:.4f}")
print(f"相对RMSE            | {r22[1]:.4f}      | {r24_256[1]:.4f}     | {r24_1024[1]:.4f}")
print(f"KE相对误差          | {r22[2]:.4f}      | {r24_256[2]:.4f}     | {r24_1024[2]:.4f}")
print(f"绝对环量误差        | {r22[3]:.4f}      | {r24_256[3]:.4f}     | {r24_1024[3]:.4f}")
print(f"最大涡量相对误差    | {r22[4]:.4f}      | {r24_256[4]:.4f}     | {r24_1024[4]:.4f}")
print(f"平均散度误差        | ~0           | {r24_256[5]:.4f}     | {r24_1024[5]:.4f}")

# Convergence: 2022 spatial order
errs_res = [m22(r, 256)[0] for r in [32, 64, 128]]
order_spatial = np.log(errs_res[0]/errs_res[1])/np.log(2)
print(f"\n2022 spatial order (32->64): {order_spatial:.2f}")
print(f"2022 RMSE res [32,64,128]: {[f'{e:.3f}' for e in errs_res]}")

# Convergence: samples
errs_nmc = [m22(64, n)[0] for n in [64, 256, 1024]]
print(f"2022 RMSE nmc [64,256,1024]: {[f'{e:.3f}' for e in errs_nmc]}")

np.savez('report_results/comprehensive.npz',
         r22=np.array(r22), r24_256=np.array(r24_256), r24_1024=np.array(r24_1024))

# Figure: metrics bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = ['RMSE', 'KE error', 'Max vort\n error']
vals22 = [r22[0], r22[2], r22[4]]
vals24 = [r24_256[0], r24_256[2], r24_256[4]]
x = np.arange(3)
w = 0.35
axes[0].bar(x-w/2, vals22, w, label='2022 (256)', color='#4C72B0')
axes[0].bar(x+w/2, vals24, w, label='2024 (256)', color='#C44E52')
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics)
axes[0].set_ylabel('Error'); axes[0].legend(); axes[0].set_title('Error metrics')
for xi, (a, b) in enumerate(zip(vals22, vals24)):
    axes[0].text(xi-w/2, a, f'{a:.2f}', ha='center', va='bottom', fontsize=9)
    axes[0].text(xi+w/2, b, f'{b:.2f}', ha='center', va='bottom', fontsize=9)

# sample convergence
axes[1].loglog([256,512,1024], [r24_256[0], 1.0626, r24_1024[0]], 'r-s', label='2024', linewidth=2)
axes[1].axhline(r22[0], color='b', ls='--', label=f'2022 platform ({r22[0]:.2f})', linewidth=2)
axes[1].set_xlabel('MC samples'); axes[1].set_ylabel('RMSE'); axes[1].legend()
axes[1].set_title('2024 sample convergence')
axes[1].grid(True, alpha=0.3)

# resolution convergence 2022
axes[2].semilogy([32,64,128], errs_res, 'b-o', label='2022', linewidth=2)
axes[2].set_xlabel('Grid resolution'); axes[2].set_ylabel('RMSE')
axes[2].legend(); axes[2].set_title('2022 resolution convergence')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_results/fig_comprehensive_metrics.png', dpi=150)
plt.close()
print("Saved fig_comprehensive_metrics.png")
