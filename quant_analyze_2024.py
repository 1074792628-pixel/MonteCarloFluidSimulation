"""
Quantitative error analysis: 2024 velocity method (FluidSimWoB).
Taylor-Green inviscid: analytical vorticity w(t) = 2*pi*cos(pi x)cos(pi y), KE conserved.

Error sources to isolate:
  1. WoB projection MC error -> vary n_part/n_rays
  2. Semi-Lagrangian interpolation error -> vary resolution
  3. Projection (divergence) error -> measure div after projection
"""
import sys, os
import numpy as np
sys.path.insert(0, '.')
from fluid_sim_wob import FluidSimWoB

def tg_vorticity_analytical(X, Y, t, nu=0.0):
    decay = np.exp(-2 * np.pi**2 * nu * t)
    return 2 * np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y) * decay

def curl2d(u, v, h):
    res = u.shape[0]
    w = np.zeros_like(u)
    w[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*h) - (u[1:-1, 2:] - u[1:-1, :-2]) / (2*h)
    return w

def run_2024(res, n_part, n_rays, steps, dt=0.05, smooth=0.3, nu=0.05):
    xs = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    vel0 = np.zeros((res, res, 2))
    vel0[..., 0] = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    vel0[..., 1] = np.sin(np.pi * X) * np.cos(np.pi * Y)
    sim = FluidSimWoB(res, nu, dt, n_particular=n_part, n_rays=n_rays, smooth_sigma=smooth)
    sim.set_velocity(vel0.copy())
    h = sim.h
    errors, kes, divs = [], [], []
    for s in range(steps):
        sim.step()
        w_num = curl2d(sim.velocity[..., 0], sim.velocity[..., 1], h)
        w_exact = tg_vorticity_analytical(X, Y, (s+1)*dt, nu)
        errors.append(np.sqrt(np.mean((w_num - w_exact)**2)))
        kes.append(0.5 * np.sum(sim.velocity**2))
        divs.append(sim.divergence_error())
    return errors, kes, divs

print("=== 2024 error vs samples (fixed res=64) ===")
for n_part, n_rays in [(32, 16), (128, 64), (512, 128), (2048, 256)]:
    errors, kes, divs = run_2024(64, n_part, n_rays, 20)
    print(f"  part={n_part:5d} ray={n_rays:4d}: vort RMSE={errors[-1]:.4f}, KE ratio={kes[-1]/kes[0]:.4f}, div={np.mean(divs):.5f}")

print("=== 2024 error vs resolution (fixed 512p+128r) ===")
for res in [16, 32, 64, 128]:
    errors, kes, divs = run_2024(res, 512, 128, 20)
    print(f"  res={res:4d}: vort RMSE={errors[-1]:.4f}, KE ratio={kes[-1]/kes[0]:.4f}, div={np.mean(divs):.5f}")
