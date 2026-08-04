"""
Additional quantitative metrics for 2022 vs 2024 comparison.
Viscous Taylor-Green (nu=0.05), res=64, t=1.0.
Measures: circulation error, KE relative error, relative RMSE,
resolution convergence (spatial order), sample convergence (MC order).
"""
import sys, os
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

# ============ 2022 extra metrics ============
def metrics_2022(res, nmc, steps=20, nu=0.05):
    sim = Sim2022GPU(res, res, 0.05, nmc, nu=nu)
    sim._init_taylor_green()
    xs = (np.arange(res)+0.5)*sim.dx + sim.ox
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    h = sim.dx
    w0 = 2*np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y)
    for s in range(steps):
        sim.step()
    w = sim.vorticity
    t = steps*0.05
    wa = w0*np.exp(-2*np.pi**2*nu*t)
    # RMSE
    rmse = np.sqrt(np.mean((w-wa)**2))
    # relative RMSE
    rel_rmse = rmse / np.abs(w0).max()
    # KE
    ke_num = 0.5*np.sum(w**2)*h*h
    ke_ana = 0.5*np.sum(wa**2)*h*h
    ke_err = abs(ke_num-ke_ana)/ke_ana
    # circulation (sum of w * area)
    circ_num = np.sum(w)*h*h
    circ_ana = np.sum(wa)*h*h
    circ_err = abs(circ_num-circ_ana)/abs(circ_ana)
    # max vorticity error
    maxw_err = abs(w.max()-wa.max())/abs(wa.max())
    return rmse, rel_rmse, ke_err, circ_err, maxw_err

print("=== 2022 metrics (res=64, nmc=256, t=1.0) ===")
m22 = metrics_2022(64, 256)
names = ['RMSE', '相对RMSE', 'KE相对误差', '环量相对误差', '最大涡量相对误差']
for n, v in zip(names, m22):
    print(f"  {n}: {v:.4f}")

# 2022 convergence with resolution (spatial order)
print("\n=== 2022 resolution convergence (nmc=256) ===")
errs = []
res_list = [16, 32, 64]
for r in res_list:
    errs.append(metrics_2022(r, 256)[0])
for r, e in zip(res_list, errs):
    print(f"  res={r}: RMSE={e:.4f}")
# order between 32 and 64
if len(errs) >= 3:
    order = np.log(errs[-2]/errs[-1])/np.log(2)
    print(f"  spatial order (32->64): {order:.2f}")

# 2022 convergence with samples (MC order)
print("\n=== 2022 sample convergence (res=64) ===")
nmcs = [64, 256, 1024]
mc_errs = [metrics_2022(64, n)[0] for n in nmcs]
for n, e in zip(nmcs, mc_errs):
    print(f"  nmc={n}: RMSE={e:.4f}")

# ============ 2024 metrics (from VelMCFluids output) ============
def read_vel(path):
    with open(path, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 2)
def curl(u, v, h, res):
    w = np.zeros((res, res))
    w[1:-1, 1:-1] = (v[1:-1,2:]-v[1:-1,:-2])/(2*h) - (u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    return w

res = 64; L = 2.4; h = L/res
xs = np.linspace(-L/2+h/2, L/2-h/2, res)
X, Y = np.meshgrid(xs, xs, indexing='xy')
k = 2*np.pi/L
t = 1.0
wa24 = 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*t)

print("\n=== 2024 metrics (res=64, samples=256, t=1.0) ===")
dirs = {256:'results_tg_compare', 1024:'results_tg_s1024'}
for s, dn in dirs.items():
    vel = read_vel(f'VelMCFluids/results/{dn}/raw/velocity_20.vector')
    u = vel[:,0].reshape(res,res); v = vel[:,1].reshape(res,res)
    w = curl(u, v, h, res)
    rmse = np.sqrt(np.mean((w-wa24)**2))
    rel = rmse / (2*k)
    ke_num = 0.5*np.sum(u**2+v**2)*h*h
    ke_ana = 0.5*np.sum((2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*t))**2)*h*h
    ke_err = abs(ke_num-ke_ana)/ke_ana
    circ_num = np.sum(w)*h*h
    circ_ana = np.sum(wa24)*h*h
    circ_err = abs(circ_num-circ_ana)/abs(circ_ana)
    maxw_err = abs(w.max()-wa24.max())/abs(wa24.max())
    print(f"  samples={s}: RMSE={rmse:.4f}, 相对RMSE={rel:.4f}, KE相对误差={ke_err:.4f}, 环量误差={circ_err:.4f}, 最大涡量误差={maxw_err:.4f}")
