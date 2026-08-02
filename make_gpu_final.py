"""
Final GPU vs GPU comparison figure:
- Error vs samples for 2022 (nmc) and 2024 (path samples)
- Timing comparison
- Error evolution over time
Viscous Taylor-Green (nu=0.05), res=64.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

# --- 2022 GPU errors at different nmc (res=64, nu=0.05, t=1.0) ---
def err22_at(nmc):
    sim = Sim2022GPU(64, 64, 0.05, nmc, nu=0.05)
    sim._init_taylor_green()
    xs = (np.arange(64)+0.5)*sim.dx + sim.ox
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    w0 = 2*np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y)
    for s in range(20):
        sim.step()
    we = w0*np.exp(-2*np.pi**2*0.05*1.0)
    return np.sqrt(np.mean((sim.vorticity - we)**2))

nmc_list = [64, 256, 1024]
e22 = [err22_at(n) for n in nmc_list]
# Timing at res=64, 40 steps
import time
sim = Sim2022GPU(64, 64, 0.05, 256, nu=0.05)
sim._init_taylor_green()
t0=time.time()
for _ in range(40): sim.step()
t22 = time.time()-t0

# --- 2024 GPU errors (from saved runs) ---
def read_vel(path):
    with open(path,'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1,2)
def curl(u,v,h,res):
    w=np.zeros((res,res)); w[1:-1,1:-1]=(v[1:-1,2:]-v[1:-1,:-2])/(2*h)-(u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    return w
res=64; L=2.4; h=L/res
xs=np.linspace(-L/2+h/2,L/2-h/2,res)
X,Y=np.meshgrid(xs,xs,indexing='xy')
k=2*np.pi/L
we24=2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*1.0)
samp_list=[256,512,1024]
dirs={256:'results_tg_compare',512:'results_tg_s512',1024:'results_tg_s1024'}
e24=[]
for s in samp_list:
    vel=read_vel(f'VelMCFluids/results/{dirs[s]}/raw/velocity_20.vector')
    u=vel[:,0].reshape(res,res); v=vel[:,1].reshape(res,res)
    w=curl(u,v,h,res)
    e24.append(np.sqrt(np.mean((w-we24)**2)))
t24 = 4.84  # from earlier timing at res=64, 40 steps, 256 samples

np.savez('report_results/gpu_error.npz', nmc_list=nmc_list, e22=np.array(e22),
         samp_list=samp_list, e24=np.array(e24))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Error vs samples
axes[0].loglog(nmc_list, e22, 'b-o', label='2022 (nmc)', linewidth=2, markersize=7)
axes[0].loglog(samp_list, e24, 'r-s', label='2024 (path samples)', linewidth=2, markersize=7)
axes[0].set_xlabel('MC samples'); axes[0].set_ylabel('Vorticity RMSE (t=1.0)')
axes[0].legend(); axes[0].set_title('Error vs MC samples')
axes[0].grid(True, alpha=0.3)

# Timing
methods = ['2022 GPU\n(CuPy)', '2024 GPU\n(OptiX)']
times = [t22, t24]
bars = axes[1].bar(methods, times, color=['#4C72B0', '#C44E52'], width=0.5)
axes[1].set_ylabel('Time (s)'); axes[1].set_title('40 steps, res=64, ~256 samples')
for b, t in zip(bars, times):
    axes[1].text(b.get_x()+b.get_width()/2, t, f'{t:.2f}s', ha='center', va='bottom', fontsize=11)
axes[1].set_ylim(0, max(times)*1.2)

# Error evolution over time (both at ~256 samples)
sim = Sim2022GPU(64, 64, 0.05, 256, nu=0.05)
sim._init_taylor_green()
xs2 = (np.arange(64)+0.5)*sim.dx + sim.ox
X2, Y2 = np.meshgrid(xs2, xs2, indexing='ij')
w02 = 2*np.pi*np.cos(np.pi*X2)*np.cos(np.pi*Y2)
err22_t = []
for s in range(20):
    sim.step()
    we = w02*np.exp(-2*np.pi**2*0.05*(s+1)*0.05)
    err22_t.append(np.sqrt(np.mean((sim.vorticity-we)**2)))
err24_t = []
for s in range(20):
    vel = read_vel(f'VelMCFluids/results/{dirs[256]}/raw/velocity_{s+1}.vector')
    u = vel[:,0].reshape(res,res); v = vel[:,1].reshape(res,res)
    w = curl(u,v,h,res)
    we = 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*(s+1)*0.05)
    err24_t.append(np.sqrt(np.mean((w-we)**2)))
t = (np.arange(20)+1)*0.05
axes[2].semilogy(t, err22_t, 'b-o', label='2022 GPU', linewidth=2)
axes[2].semilogy(t, err24_t, 'r-s', label='2024 GPU', linewidth=2)
axes[2].set_xlabel('Time t'); axes[2].set_ylabel('Vorticity RMSE')
axes[2].legend(); axes[2].set_title('Error evolution (256 samples)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_results/gpu_vs_gpu_final.png', dpi=150)
plt.close()

print("=== Final GPU vs GPU summary ===")
print(f"2022 GPU timing (40 steps, res=64): {t22:.2f}s")
print(f"2024 GPU timing (40 steps, res=64): {t24:.2f}s")
print(f"2022 error: nmc {nmc_list} -> RMSE {[f'{e:.3f}' for e in e22]}")
print(f"2024 error: samples {samp_list} -> RMSE {[f'{e:.3f}' for e in e24]}")
print("Saved report_results/gpu_vs_gpu_final.png")
