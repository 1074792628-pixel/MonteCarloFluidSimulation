#!/usr/bin/env python3
"""
Comparison: Pressure Poisson (2022) vs Stream Function (2022 boundary method)

Pressure:  u → advect → diffuse(WoS) → ∇·u → ∇²p=∇·u(WoS) → u=u*-∇p
Stream fn: u → advect → diffuse(WoS) → ω=∇×u → ∇²ψ=-ω(WoS) → u=∇×ψ

The stream function approach is derived from the 2022 paper's boundary handling,
NOT from the 2024 "Velocity-Based Monte Carlo Fluids" paper.

Test cases:
  - taylor : Taylor-Green decaying vortices
  - cylinder: Flow past a cylinder (Karman vortex street)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import argparse

from wos_core import rect_obstacle_domain, curl_2d, taylor_green_vortex
from fluid_sim_2022 import FluidSim2022
from fluid_sim_2024 import FluidSimStreamFn


# ─── Domain ──────────────────────────────────────────────────────────────

def make_empty_domain():
    """Square [-1,1]x[-1,1] with a dummy obstacle far away."""
    return rect_obstacle_domain(rect_size=2.0, obs_center=(99, 99), obs_radius=0.01)


def make_cylinder_domain():
    """Square [-1,1]x[-1,1] with a cylinder at origin."""
    return rect_obstacle_domain(rect_size=2.0, obs_center=(0.0, 0.0), obs_radius=0.15)


# ─── Initial Conditions ──────────────────────────────────────────────────

def init_flow_past_cylinder(res, domain, u_inflow=0.8):
    X, Y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing='ij')
    vel = np.zeros((res, res, 2))
    vel[..., 0] = u_inflow
    inside = domain.inside(np.stack([X, Y], axis=-1).reshape(-1, 2)).reshape(res, res)
    vel[~inside] = 0.0
    return vel


# ─── Visualization ──────────────────────────────────────────────────────

def save_comparison_frame(sim22, sim_sf, step, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, (sim, name) in enumerate([(sim22, 'Pressure Poisson (2022)'),
                                        (sim_sf, 'Stream Function (2022 boundary method)')]):
        curl = sim.vorticity()
        speed = np.sqrt(sim.velocity[..., 0]**2 + sim.velocity[..., 1]**2)
        vmax = max(np.abs(curl).max(), 1e-6)

        axes[row, 0].imshow(curl.T, origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
        axes[row, 0].set_title(f'{name}\nVorticity')
        axes[row, 0].set_xlabel('x')

        im = axes[row, 1].imshow(speed.T, origin='lower', cmap='viridis',
                                  extent=[-1, 1, -1, 1])
        axes[row, 1].set_title('Speed |u|')
        axes[row, 1].set_xlabel('x')
        plt.colorbar(im, ax=axes[row, 1])

        skip = max(1, sim.res // 16)
        axes[row, 2].quiver(sim.X[::skip, ::skip], sim.Y[::skip, ::skip],
                            sim.velocity[::skip, ::skip, 0],
                            sim.velocity[::skip, ::skip, 1],
                            scale=1.5, width=0.005)
        axes[row, 2].set_title(f'Vector field\nt={sim.time:.2f}')
        axes[row, 2].set_xlim(-1, 1)
        axes[row, 2].set_ylim(-1, 1)
        axes[row, 2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'comparison_{step:04d}.png'), dpi=120)
    plt.close()


def save_summary_plots(e22, e_sf, d22, d_sf, sim22, sim_sf, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(e22, label='Pressure Poisson', marker='.')
    axes[0].plot(e_sf, label='Stream Function', marker='.')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Kinetic Energy')
    axes[0].legend()
    axes[0].set_title('Energy Decay')

    axes[1].plot(d22, label='Pressure Poisson', marker='.')
    axes[1].plot(d_sf, label='Stream Function', marker='.')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Mean |div u|')
    axes[1].legend()
    axes[1].set_title('Divergence Error (lower=better)')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'energy_divergence.png'), dpi=120)
    plt.close()

    # Final comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    curl22 = sim22.vorticity()
    curl_sf = sim_sf.vorticity()
    vmax = max(np.abs(curl22).max(), np.abs(curl_sf).max(), 1e-6)

    im0 = axes[0].imshow(curl22.T, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
    axes[0].set_title('Pressure Poisson Vorticity')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(curl_sf.T, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
    axes[1].set_title('Stream Function Vorticity')
    plt.colorbar(im1, ax=axes[1])

    diff = np.abs(curl22) - np.abs(curl_sf)
    dmax = max(np.abs(diff).max(), 1e-6)
    im2 = axes[2].imshow(diff.T, origin='lower', cmap='RdBu_r',
                          vmin=-dmax, vmax=dmax, extent=[-1, 1, -1, 1])
    axes[2].set_title('|ω_pressure| - |ω_streamfn|')
    plt.colorbar(im2, ax=axes[2])

    div22 = (sim22.velocity[2:, 1:-1, 0] - sim22.velocity[:-2, 1:-1, 0]) / (2*sim22.h) + \
            (sim22.velocity[1:-1, 2:, 1] - sim22.velocity[1:-1, :-2, 1]) / (2*sim22.h)
    div_sf = (sim_sf.velocity[2:, 1:-1, 0] - sim_sf.velocity[:-2, 1:-1, 0]) / (2*sim_sf.h) + \
             (sim_sf.velocity[1:-1, 2:, 1] - sim_sf.velocity[1:-1, :-2, 1]) / (2*sim_sf.h)
    dvmax = max(np.abs(div22).max(), np.abs(div_sf).max(), 1e-6)
    axes[3].semilogy(np.abs(div22).ravel(), '.', alpha=0.3, label='Pressure Poisson')
    axes[3].semilogy(np.abs(div_sf).ravel(), '.', alpha=0.3, label='Stream Function')
    axes[3].legend()
    axes[3].set_title('Divergence |∇·u| per cell')
    axes[3].set_ylabel('|∇·u|')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'final_comparison.png'), dpi=150)
    plt.close()


# ─── Runner ──────────────────────────────────────────────────────────────

def run_test(test_name, grid_res, n_walks, n_steps, dt, nu, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if test_name == 'taylor':
        domain = make_empty_domain()
        vel0 = taylor_green_vortex(grid_res, 0.0, nu)
    elif test_name == 'cylinder':
        domain = make_cylinder_domain()
        vel0 = init_flow_past_cylinder(grid_res, domain, u_inflow=0.8)
    else:
        raise ValueError(f"Unknown test: {test_name}")

    sim22 = FluidSim2022(domain, grid_res, nu, dt, n_walks)
    sim_sf = FluidSimStreamFn(domain, grid_res, nu, dt, n_walks)
    sim22.set_velocity(vel0.copy())
    sim_sf.set_velocity(vel0.copy())

    e22, e_sf = [], []
    d22, d_sf = [], []

    print(f"\nTest: {test_name} | Grid: {grid_res}x{grid_res} | Walks: {n_walks} | Steps: {n_steps}")
    print(f"{'Step':>5} | {'KE_Press':>9} {'KE_StrFn':>9} | {'Div_Press':>9} {'Div_StrFn':>9} | {'Time':>7}")
    print("-" * 60)

    for step in range(n_steps):
        ta = time.time()
        sim22.step()
        tb = time.time()
        sim_sf.step()
        tc = time.time()

        e22.append(sim22.kinetic_energy())
        e_sf.append(sim_sf.kinetic_energy())
        d22.append(sim22.divergence_error())
        d_sf.append(sim_sf.divergence_error())

        if step % 5 == 0 or step == n_steps - 1:
            print(f"{step:5d} | {e22[-1]:9.4f} {e_sf[-1]:9.4f} | {d22[-1]:9.6f} {d_sf[-1]:9.6f} | {(tc-ta):6.1f}s")
            save_comparison_frame(sim22, sim_sf, step, out_dir)

    save_summary_plots(e22, e_sf, d22, d_sf, sim22, sim_sf, out_dir)

    print(f"\n=== Summary ===")
    print(f"Pressure Poisson: Final KE={e22[-1]:.4f}, Mean|div|={np.mean(d22):.6f}, Time={sim22.stats['total_time']:.1f}s")
    print(f"Stream Function:  Final KE={e_sf[-1]:.4f}, Mean|div|={np.mean(d_sf):.6f}, Time={sim_sf.stats['total_time']:.1f}s")
    print(f"2022 calls: {sim22.stats['pressure_solves']} pressure + {sim22.stats['diffusion_steps']} diffusion")
    print(f"Stream fn calls: {sim_sf.stats['stream_function_solves']} stream fn + {sim_sf.stats['diffusion_steps']} diffusion")

    np.savez(os.path.join(out_dir, 'results.npz'),
             vel22=sim22.velocity, vel_sf=sim_sf.velocity,
             curl22=sim22.vorticity(), curl_sf=sim_sf.vorticity(),
             e22=e22, e_sf=e_sf, d22=d22, d_sf=d_sf)

    return sim22, sim_sf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Pressure Poisson vs Stream Function Monte Carlo Fluids')
    parser.add_argument('--grid', type=int, default=16, help='Grid resolution')
    parser.add_argument('--walks', type=int, default=128, help='MC walks per point')
    parser.add_argument('--steps', type=int, default=20, help='Number of time steps')
    parser.add_argument('--dt', type=float, default=0.03, help='Time step')
    parser.add_argument('--nu', type=float, default=0.05, help='Viscosity')
    parser.add_argument('--test', choices=['taylor', 'cylinder'], default='taylor')
    parser.add_argument('--out', type=str, default='comparison_results')
    args = parser.parse_args()

    print("=" * 60)
    print("Monte Carlo Fluids: Pressure Poisson vs Stream Function")
    print("=" * 60)

    run_test(args.test, args.grid, args.walks, args.steps,
             args.dt, args.nu, args.out)

    print(f"\nResults -> '{args.out}/'")
