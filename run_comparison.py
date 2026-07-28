#!/usr/bin/env python3
"""
Comparison: 2022 (WoS pressure-Poisson) vs 2024 (Stream function) Monte Carlo Fluids

2022: u → advect → diffuse(WoS) → ∇·u → ∇²p=∇·u(WoS) → u=u*-∇p
2024: u → advect → diffuse(WoS) → ω=∇×u → ∇²ψ=-ω(WoS) → u=∇×ψ

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
from fluid_sim_2024 import FluidSim2024


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

def save_comparison_frame(sim22, sim24, step, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, (sim, name) in enumerate([(sim22, '2022: Pressure Poisson'),
                                        (sim24, '2024: Stream Function')]):
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


def save_summary_plots(e22, e24, d22, d24, sim22, sim24, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(e22, label='2022 (Pressure Poisson)', marker='.')
    axes[0].plot(e24, label='2024 (Stream Function)', marker='.')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Kinetic Energy')
    axes[0].legend()
    axes[0].set_title('Energy Decay')

    axes[1].plot(d22, label='2022', marker='.')
    axes[1].plot(d24, label='2024 (Stream function)', marker='.')
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
    curl24 = sim24.vorticity()
    vmax = max(np.abs(curl22).max(), np.abs(curl24).max(), 1e-6)

    im0 = axes[0].imshow(curl22.T, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
    axes[0].set_title('2022 Vorticity')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(curl24.T, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
    axes[1].set_title('2024 Vorticity')
    plt.colorbar(im1, ax=axes[1])

    diff = np.abs(curl22) - np.abs(curl24)
    dmax = max(np.abs(diff).max(), 1e-6)
    im2 = axes[2].imshow(diff.T, origin='lower', cmap='RdBu_r',
                          vmin=-dmax, vmax=dmax, extent=[-1, 1, -1, 1])
    axes[2].set_title('|ω22| - |ω24|')
    plt.colorbar(im2, ax=axes[2])

    div22 = (sim22.velocity[2:, 1:-1, 0] - sim22.velocity[:-2, 1:-1, 0]) / (2*sim22.h) + \
            (sim22.velocity[1:-1, 2:, 1] - sim22.velocity[1:-1, :-2, 1]) / (2*sim22.h)
    div24 = (sim24.velocity[2:, 1:-1, 0] - sim24.velocity[:-2, 1:-1, 0]) / (2*sim24.h) + \
            (sim24.velocity[1:-1, 2:, 1] - sim24.velocity[1:-1, :-2, 1]) / (2*sim24.h)
    dvmax = max(np.abs(div22).max(), np.abs(div24).max(), 1e-6)
    axes[3].semilogy(np.abs(div22).ravel(), '.', alpha=0.3, label='2022')
    axes[3].semilogy(np.abs(div24).ravel(), '.', alpha=0.3, label='2024')
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
    sim24 = FluidSim2024(domain, grid_res, nu, dt, n_walks)
    sim22.set_velocity(vel0.copy())
    sim24.set_velocity(vel0.copy())

    e22, e24 = [], []
    d22, d24 = [], []

    print(f"\nTest: {test_name} | Grid: {grid_res}x{grid_res} | Walks: {n_walks} | Steps: {n_steps}")
    print(f"{'Step':>5} | {'KE_2022':>9} {'KE_2024':>9} | {'Div_2022':>9} {'Div_2024':>9} | {'Time':>7}")
    print("-" * 60)

    for step in range(n_steps):
        ta = time.time()
        sim22.step()
        tb = time.time()
        sim24.step()
        tc = time.time()

        e22.append(sim22.kinetic_energy())
        e24.append(sim24.kinetic_energy())
        d22.append(sim22.divergence_error())
        d24.append(sim24.divergence_error())

        if step % 5 == 0 or step == n_steps - 1:
            print(f"{step:5d} | {e22[-1]:9.4f} {e24[-1]:9.4f} | {d22[-1]:9.6f} {d24[-1]:9.6f} | {(tc-ta):6.1f}s")
            save_comparison_frame(sim22, sim24, step, out_dir)

    save_summary_plots(e22, e24, d22, d24, sim22, sim24, out_dir)

    print(f"\n=== Summary ===")
    print(f"2022: Final KE={e22[-1]:.4f}, Mean|div|={np.mean(d22):.6f}, Time={sim22.stats['total_time']:.1f}s")
    print(f"2024: Final KE={e24[-1]:.4f}, Mean|div|={np.mean(d24):.6f}, Time={sim24.stats['total_time']:.1f}s")
    print(f"2022 calls: {sim22.stats['pressure_solves']} pressure + {sim22.stats['diffusion_steps']} diffusion")
    print(f"2024 calls: {sim24.stats['stream_function_solves']} stream fn + {sim24.stats['diffusion_steps']} diffusion")

    np.savez(os.path.join(out_dir, 'results.npz'),
             vel22=sim22.velocity, vel24=sim24.velocity,
             curl22=sim22.vorticity(), curl24=sim24.vorticity(),
             e22=e22, e24=e24, d22=d22, d24=d24)

    return sim22, sim24


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare 2022 vs 2024 Monte Carlo Fluids')
    parser.add_argument('--grid', type=int, default=16, help='Grid resolution')
    parser.add_argument('--walks', type=int, default=128, help='MC walks per point')
    parser.add_argument('--steps', type=int, default=20, help='Number of time steps')
    parser.add_argument('--dt', type=float, default=0.03, help='Time step')
    parser.add_argument('--nu', type=float, default=0.05, help='Viscosity')
    parser.add_argument('--test', choices=['taylor', 'cylinder'], default='taylor')
    parser.add_argument('--out', type=str, default='comparison_results')
    args = parser.parse_args()

    print("=" * 60)
    print("Monte Carlo Fluids: 2022 (Pressure WoS) vs 2024 (Stream Function WoS)")
    print("=" * 60)

    run_test(args.test, args.grid, args.walks, args.steps,
             args.dt, args.nu, args.out)

    print(f"\nResults -> '{args.out}/'")
