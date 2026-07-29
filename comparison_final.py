#!/usr/bin/env python3
"""
Comparison: Pressure-Poisson vs Stream Function projection for MC Fluids

Both use Numba-accelerated Walk-on-Spheres for:
  - Diffusion step
  - Poisson solve (for p or ψ)

Metrics: KE, divergence error, vorticity, runtime

Test: Taylor-Green decaying vortices

References:
  [2022] Rioux-Lavoie et al. "A Monte Carlo Method for Fluid Simulation"
"""
import numpy as np, time, os, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fluid_sim_2022_numba import FluidSim2022N
from fluid_sim_stream_numba import FluidSimStream


def taylor_green_vortex(res, nu=0.05, t=0.0):
    xs = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    decay = np.exp(-2 * np.pi**2 * nu * t)
    u = np.zeros((res, res, 2))
    u[..., 0] = -np.cos(np.pi * X) * np.sin(np.pi * Y) * decay
    u[..., 1] = np.sin(np.pi * X) * np.cos(np.pi * Y) * decay
    return u


def run_comparison(grid_res=12, n_walks=256, n_steps=20, dt=0.04, nu=0.05,
                   has_obs=False, ox=0.0, oy=0.0, orad=0.15, out_dir='results_final'):
    os.makedirs(out_dir, exist_ok=True)

    # Initialize simulations
    sim22 = FluidSim2022N(grid_res, nu, dt, n_walks, has_obs, ox, oy, orad)
    sim_sf = FluidSimStream(grid_res, nu, dt, n_walks, has_obs, ox, oy, orad)

    vel0 = taylor_green_vortex(grid_res, nu)
    if has_obs:
        xs = np.linspace(-1, 1, grid_res)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        for i in range(grid_res):
            for j in range(grid_res):
                d = np.sqrt((X[i,j]-ox)**2 + (Y[i,j]-oy)**2) - orad
                if d < 0:
                    vel0[i, j] = 0.0

    sim22.set_velocity(vel0.copy())
    sim_sf.set_velocity(vel0.copy())

    e22, e_sf = [], []
    d22, d_sf = [], []
    times = []

    print(f"\nGrid: {grid_res}x{grid_res} | Walks: {n_walks} | Steps: {n_steps}")
    print(f"{'Step':>5} | {'KE_Press':>9} {'KE_StrFn':>9} | {'Div_Press':>9} {'Div_StrFn':>9} | {'Time':>6}")
    print("-" * 60)

    for step in range(n_steps):
        t0 = time.time()
        sim22.step()
        ta = time.time()
        sim_sf.step()
        tb = time.time()
        times.append((ta - t0, tb - ta))

        e22.append(sim22.kinetic_energy())
        e_sf.append(sim_sf.kinetic_energy())
        d22.append(sim22.divergence_error())
        d_sf.append(sim_sf.divergence_error())

        if step % max(1, n_steps // 5) == 0 or step == n_steps - 1:
            print(f"{step:5d} | {e22[-1]:9.2f} {e_sf[-1]:9.2f} | {d22[-1]:9.6f} {d_sf[-1]:9.6f} | {tb-t0:6.1f}s")

    # Generate comparison figures
    _plot_results(sim22, sim_sf, e22, e_sf, d22, d_sf, out_dir)

    print(f"\n=== Summary ===")
    print(f"2022 (Pressure Poisson WoS):")
    print(f"  Final KE = {e22[-1]:.2f}, Mean|div| = {np.mean(d22):.6f}")
    print(f"  Time: {sim22.stats['total_time']:.1f}s ({sum(t[0] for t in times):.1f}s)")
    print(f"Stream fn (based on 2022 boundary handling):")
    print(f"  Final KE = {e_sf[-1]:.2f}, Mean|div| = {np.mean(d_sf):.6f}")
    print(f"  Time: {sim_sf.stats['total_time']:.1f}s ({sum(t[1] for t in times):.1f}s)")

    # Save data
    np.savez(os.path.join(out_dir, 'results.npz'),
             vel22=sim22.velocity, vel_sf=sim_sf.velocity,
             curl22=sim22.vorticity(), curl_sf=sim_sf.vorticity(),
             e22=e22, e_sf=e_sf, d22=d22, d_sf=d_sf)
    print(f"\nResults saved to '{out_dir}/'")


def _plot_results(sim22, sim_sf, e22, e_sf, d22, d_sf, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    vmax_22 = max(abs(sim22.vorticity()).max(), 1e-6)
    vmax_sf = max(abs(sim_sf.vorticity()).max(), 1e-6)
    vmax = max(vmax_22, vmax_sf)

    for row, (sim, name) in enumerate([(sim22, 'Pressure-Poisson (2022)'),
                                        (sim_sf, 'Stream Function (2022 boundary method)')]):
        axes[row, 0].imshow(sim.vorticity().T, origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
        axes[row, 0].set_title(f'{name}\nVorticity')

        speed = np.sqrt(sim.velocity[..., 0]**2 + sim.velocity[..., 1]**2)
        im = axes[row, 1].imshow(speed.T, origin='lower', cmap='viridis',
                                  extent=[-1, 1, -1, 1])
        axes[row, 1].set_title('Speed |u|')
        plt.colorbar(im, ax=axes[row, 1])

        skip = max(1, sim.res // 12)
        axes[row, 2].quiver(sim.X[::skip, ::skip], sim.Y[::skip, ::skip],
                            sim.velocity[::skip, ::skip, 0],
                            sim.velocity[::skip, ::skip, 1],
                            scale=2.0, width=0.005)
        axes[row, 2].set_title('Vector field')
        axes[row, 2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'final_fields.png'), dpi=120)

    # Energy & divergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(e22, 'b.-', label='2022 (Pressure)')
    axes[0].plot(e_sf, 'r.-', label='Stream fn')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('KE'); axes[0].legend()
    axes[0].set_title('Kinetic Energy')

    axes[1].plot(d22, 'b.-', label='2022')
    axes[1].plot(d_sf, 'r.-', label='Stream fn')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Mean |div u|')
    axes[1].legend(); axes[1].set_title('Divergence Error')
    axes[1].set_yscale('log')

    # Side-by-side vorticity at final time
    curl22 = sim22.vorticity()
    curl_sf = sim_sf.vorticity()
    diff = abs(curl22) - abs(curl_sf)
    dmax = max(abs(diff).max(), 1e-6)
    im = axes[2].imshow(diff.T, origin='lower', cmap='RdBu_r',
                         vmin=-dmax, vmax=dmax, extent=[-1, 1, -1, 1])
    axes[2].set_title('|ω_pressure| - |ω_streamfn|')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'energy_divergence.png'), dpi=120)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pressure Poisson vs Stream Function Monte Carlo Fluids')
    parser.add_argument('--grid', type=int, default=10, help='Grid resolution')
    parser.add_argument('--walks', type=int, default=256, help='MC walks per point')
    parser.add_argument('--steps', type=int, default=10, help='Time steps')
    parser.add_argument('--dt', type=float, default=0.03, help='Time step')
    parser.add_argument('--nu', type=float, default=0.05, help='Viscosity')
    parser.add_argument('--obs', action='store_true', help='Cylinder obstacle')
    parser.add_argument('--out', type=str, default='comparison_final')
    args = parser.parse_args()

    print("=" * 55)
    print("MC Fluids: Pressure Poisson (2022) vs Stream Function (2022 boundary method)")
    print("Using Numba-accelerated Walk-on-Spheres")
    print("=" * 55)

    obj_name = " with obstacle" if args.obs else ""
    print(f"Test case: Taylor-Green vortices{obj_name}")

    run_comparison(args.grid, args.walks, args.steps, args.dt, args.nu,
                   args.obs, 0.0, 0.0, 0.15, args.out)
