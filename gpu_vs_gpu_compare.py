"""
GPU vs GPU quantitative comparison: 2022 (CuPy vorticity) vs 2024 (OptiX velocity).
Both on Taylor-Green with their respective domain scaling, viscous (nu=0.05).
Reads 2024 GPU output from VelMCFluids, computes vorticity error vs analytical.
Runs 2022 GPU with matching setup.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')

# ---- Read 2024 GPU output ----
def read_vector(path, dim):
    with open(path, 'rb') as f:
        grid = np.frombuffer(f.read(2 * 4), dtype=np.int32)
        data = np.frombuffer(f.read(), dtype=np.float32)
    return grid, data.reshape(-1, dim)

def curl_from_vel(u, v, h, res):
    # u,v are (res,res) with u[y,x] = u(x,y) (y-major storage)
    # w = dv/dx - du/dy = (axis1 of v) - (axis0 of u)
    w = np.zeros((res, res))
    w[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2*h) - (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*h)
    return w

def tg_w_2024(X, Y, t, nu, L=2.4):
    k = 2*np.pi/L
    return 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*nu*t)

res = 64
L = 2.4
h = L / res
xs = np.linspace(-L/2 + h/2, L/2 - h/2, res)
# 'xy' indexing: X[a,b]=x[b], Y[a,b]=y[a], matching u[y,x] storage
X, Y = np.meshgrid(xs, xs, indexing='xy')

raw = 'VelMCFluids/results/results_tg_compare/raw'
errors24 = []
kes24 = []
for step in range(1, 21):  # t=0.05 to 1.0
    f = f'{raw}/velocity_{step}.vector'
    if not os.path.exists(f):
        break
    grid, vel = read_vector(f, 2)
    u = vel[:, 0].reshape(res, res)  # u[y,x]
    v = vel[:, 1].reshape(res, res)
    w_num = curl_from_vel(u, v, h, res)
    w_exact = tg_w_2024(X, Y, step*0.05, 0.05, L)
    errors24.append(np.sqrt(np.mean((w_num - w_exact)**2)))
    kes24.append(0.5*np.sum(vel[:, 0]**2 + vel[:, 1]**2))

print(f"2024 GPU (OptiX): final vort RMSE={errors24[-1]:.4f}, KE ratio={kes24[-1]/kes24[0]:.4f}")

# ---- Run 2022 GPU matching ----
from gpu_fluid_2022 import Sim2022GPU
sim22 = Sim2022GPU(res, res, 0.05, 256, nu=0.05)
sim22._init_taylor_green()
xs2 = (np.arange(res)+0.5)*sim22.dx + sim22.ox
X2, Y2 = np.meshgrid(xs2, xs2, indexing='ij')
w0 = 2*np.pi*np.cos(np.pi*X2)*np.cos(np.pi*Y2)
errors22 = []
kes22 = []
for s in range(20):
    sim22.step()
    w_exact = w0*np.exp(-2*np.pi**2*0.05*(s+1)*0.05)
    errors22.append(np.sqrt(np.mean((sim22.vorticity - w_exact)**2)))
    kes22.append(0.5*np.sum(sim22.vorticity**2))
print(f"2022 GPU (CuPy): final vort RMSE={errors22[-1]:.4f}, KE ratio={kes22[-1]/kes22[0]:.4f}")

# Save
np.savez('report_results/gpu_vs_gpu.npz',
         errors22=np.array(errors22), kes22=np.array(kes22),
         errors24=np.array(errors24), kes24=np.array(kes24))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
t = np.arange(1, len(errors22)+1)*0.05
axes[0].semilogy(t, errors22, 'b-o', label='2022 GPU (CuPy)', linewidth=2)
axes[0].semilogy(t, errors24, 'r-s', label='2024 GPU (OptiX)', linewidth=2)
axes[0].set_xlabel('Time t'); axes[0].set_ylabel('Vorticity RMSE vs analytical')
axes[0].legend(); axes[0].set_title('Error evolution (nu=0.05, res=64)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, kes22/np.max(kes22), 'b-o', label='2022 GPU', linewidth=2)
axes[1].plot(t, kes24/np.max(kes24), 'r-s', label='2024 GPU', linewidth=2)
axes[1].set_xlabel('Time t'); axes[1].set_ylabel('Normalized KE')
axes[1].legend(); axes[1].set_title('KE evolution')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_results/gpu_vs_gpu.png', dpi=150)
plt.close()
print("Saved report_results/gpu_vs_gpu.png")
