"""
Numba-accelerated comparison: 2022 vs 2024 Monte Carlo Fluids
"""
import numpy as np, time, os, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fluid_sim_2022_numba import FluidSim2022N
from fluid_sim_2024_numba import FluidSim2024N
from numba_wos import curl_2d_numba, divergence_2d_numba
from scipy.ndimage import gaussian_filter as gs


def run_test(grid_res, n_walks, n_steps, dt, nu, out_dir, has_obs=False,
             ox=0.0, oy=0.0, orad=0.15):
    os.makedirs(out_dir, exist_ok=True)
    xs = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(xs, xs, indexing='ij')

    # Taylor-Green initial condition
    vel0 = np.zeros((grid_res, grid_res, 2))
    vel0[..., 0] = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    vel0[..., 1] = np.sin(np.pi * X) * np.cos(np.pi * Y)
    if has_obs:
        for i in range(grid_res):
            for j in range(grid_res):
                d = np.sqrt((X[i,j]-ox)**2 + (Y[i,j]-oy)**2) - orad
                if d < 0:
                    vel0[i, j] = 0.0

    sim22 = FluidSim2022N(grid_res, nu, dt, n_walks, has_obs, ox, oy, orad)
    sim24 = FluidSim2024N(grid_res, nu, dt, n_walks, has_obs, ox, oy, orad)
    sim22.set_velocity(vel0.copy())
    sim24.set_velocity(vel0.copy())

    e22, e24 = [], []
    d22, d24 = [], []

    print(f"\nGrid: {grid_res}x{grid_res} | Walks: {n_walks} | Steps: {n_steps} | dt={dt} | nu={nu}")
    print(f"{'Step':>5} | {'KE22':>8} {'KE24':>8} | {'Div22':>9} {'Div24':>9} | Time")
    print("-" * 55)

    for step in range(n_steps):
        t0 = time.time()
        sim22.step()
        t22 = time.time() - t0
        sim24.step()
        t24 = time.time() - t22 - t0

        e22.append(sim22.kinetic_energy())
        e24.append(sim24.kinetic_energy())
        d22.append(sim22.divergence_error())
        d24.append(sim24.divergence_error())

        if step % 5 == 0 or step == n_steps - 1:
            print(f"{step:5d} | {e22[-1]:8.2f} {e24[-1]:8.2f} | {d22[-1]:9.6f} {d24[-1]:9.6f} | {t22+t24:.1f}s")

    # Final visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (sim, name) in enumerate([(sim22, '2022: Pressure Poisson'),
                                        (sim24, '2024: Stream Function')]):
        curl = sim.vorticity()
        speed = np.sqrt(sim.velocity[..., 0]**2 + sim.velocity[..., 1]**2)
        vmax = max(abs(curl).max(), 1e-6)

        axes[row, 0].imshow(curl.T, origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
        axes[row, 0].set_title(f'{name} - Vorticity')

        im = axes[row, 1].imshow(speed.T, origin='lower', cmap='viridis',
                                  extent=[-1, 1, -1, 1])
        axes[row, 1].set_title('Speed')
        plt.colorbar(im, ax=axes[row, 1])

        skip = max(1, grid_res // 12)
        axes[row, 2].quiver(sim.X[::skip, ::skip], sim.Y[::skip, ::skip],
                            sim.velocity[::skip, ::skip, 0],
                            sim.velocity[::skip, ::skip, 1],
                            scale=1.5, width=0.005)
        axes[row, 2].set_title('Vector field')
        axes[row, 2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'final_comparison.png'), dpi=120)
    plt.close()

    # Energy and divergence plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(e22, label='2022 (Pressure)', marker='.')
    axes[0].plot(e24, label='2024 (Stream fn)', marker='.')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('KE'); axes[0].legend()
    axes[0].set_title('Kinetic Energy')

    axes[1].plot(d22, label='2022', marker='.')
    axes[1].plot(d24, label='2024', marker='.')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Mean |div|')
    axes[1].legend(); axes[1].set_title('Divergence Error')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'energy_divergence.png'), dpi=120)
    plt.close()

    np.savez(os.path.join(out_dir, 'results.npz'),
             vel22=sim22.velocity, vel24=sim24.velocity,
             e22=e22, e24=e24, d22=d22, d24=d24)

    print(f"\n=== Summary ===")
    print(f"2022: Final KE={e22[-1]:.2f}, Mean|div|={np.mean(d22):.6f}")
    print(f"2024: Final KE={e24[-1]:.2f}, Mean|div|={np.mean(d24):.6f}")
    print(f"2022 time: {sim22.stats['total_time']:.1f}s, 2024 time: {sim24.stats['total_time']:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=12, help='Grid res')
    parser.add_argument('--walks', type=int, default=256, help='MC walks')
    parser.add_argument('--steps', type=int, default=20, help='Time steps')
    parser.add_argument('--dt', type=float, default=0.04, help='Time step')
    parser.add_argument('--nu', type=float, default=0.05, help='Viscosity')
    parser.add_argument('--obs', action='store_true', help='Add obstacle')
    parser.add_argument('--out', type=str, default='results_numba')
    args = parser.parse_args()

    print("=" * 50)
    print("Monte Carlo Fluids: 2022 vs 2024 (Numba)")
    print("=" * 50)
    run_test(args.grid, args.walks, args.steps, args.dt, args.nu,
             args.out, has_obs=args.obs)
