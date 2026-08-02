"""
Quantitative error analysis: 2022 vorticity method (GPU).
Taylor-Green inviscid: analytical vorticity w(t) = 2*pi*cos(pi x)cos(pi y), KE conserved.

Error sources to isolate:
  1. Biot-Savart MC integration error  -> vary nmc (at fixed high resolution)
  2. Semi-Lagrangian interpolation error -> vary resolution (at fixed high nmc)
"""
import sys, os
import numpy as np
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

def tg_vorticity_analytical(X, Y, t, nu=0.0):
    decay = np.exp(-2 * np.pi**2 * nu * t)
    return 2 * np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y) * decay

def run_2022(res, nmc, steps, dt=0.05, nu=0.05):
    sim = Sim2022GPU(res, res, dt, nmc, nu=nu)
    sim._init_taylor_green()
    xs = (np.arange(res) + 0.5) * sim.dx + sim.ox
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    errors = []
    kes = []
    for s in range(steps):
        sim.step()
        w_num = sim.vorticity
        w_exact = tg_vorticity_analytical(XX, YY, (s+1)*dt, nu)
        errors.append(np.sqrt(np.mean((w_num - w_exact)**2)))
        kes.append(0.5 * np.sum(w_num**2))
    return errors, kes

print("=== 2022 GPU error vs nmc (fixed res=64, nu=0.05) ===")
for nmc in [16, 64, 256, 1024]:
    errors, kes = run_2022(64, nmc, 20)
    print(f"  nmc={nmc:5d}: vorticity RMSE final={errors[-1]:.4f}, KE ratio final={kes[-1]/kes[0]:.4f}")

print("=== 2022 GPU error vs resolution (fixed nmc=256, nu=0.05) ===")
for res in [16, 32, 64, 128]:
    errors, kes = run_2022(res, 256, 20)
    print(f"  res={res:4d}: vorticity RMSE final={errors[-1]:.4f}, KE ratio final={kes[-1]/kes[0]:.4f}")
