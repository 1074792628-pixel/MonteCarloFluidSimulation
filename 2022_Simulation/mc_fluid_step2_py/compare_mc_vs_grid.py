import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from core.simulator import Simulator
from core.geometry import Geometry
from core.types import Vec2
from core.grid_stream_solver import GridStreamFunctionSolver

# ============================================================
# 初始条件：两个高斯涡旋（避开障碍物）
# ============================================================
def init_vorticity(nx, ny, dx, ox, oy, geometry):
    grid = np.zeros((ny, nx))
    sigma = 0.2
    centers = [Vec2(-0.5, 0.0), Vec2(0.5, 0.0)]
    for j in range(ny):
        for i in range(nx):
            p = Vec2(ox + (i+0.5)*dx, oy + (j+0.5)*dx)
            if not geometry.inside_domain(p):
                continue
            w = 0.0
            for c in centers:
                d = (p - c).norm()
                w += np.exp(-d*d/(2*sigma*sigma))
            grid[j, i] = w
    return grid

def main():
    parser = argparse.ArgumentParser(description='Step2: MC vs Grid Stream Function')
    parser.add_argument('--nx', type=int, default=64, help='网格横向分辨率')
    parser.add_argument('--ny', type=int, default=64, help='网格纵向分辨率')
    parser.add_argument('--dt', type=float, default=0.05, help='时间步长')
    parser.add_argument('--total_time', type=float, default=1.0, help='总模拟时长')
    parser.add_argument('--nmc', type=int, default=64, help='MC WoS路径数')
    parser.add_argument('--obstacle_radius', type=float, default=0.3, help='障碍物半径')
    parser.add_argument('--obstacle_center_x', type=float, default=0.0)
    parser.add_argument('--obstacle_center_y', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()
    nx, ny = args.nx, args.ny
    dt, total_time = args.dt, args.total_time
    ox, oy = -1.0, -1.0
    dx = 2.0 / nx
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    geo = Geometry(outer_bounds=(-1.0, 1.0, -1.0, 1.0))
    geo.add_circle(Vec2(args.obstacle_center_x, args.obstacle_center_y), args.obstacle_radius)
    omega0 = init_vorticity(nx, ny, dx, ox, oy, geo)
    
    # ---------- MC 方法 ----------
    print("运行 MC (Step2, WoS)...")
    sim = Simulator(nx=nx, ny=ny, dt=dt, geometry=geo)
    sim.nmc = args.nmc
    for j in range(ny):
        for i in range(nx):
            sim.grid[0].set_vort(i, j, omega0[j, i])
            sim.grid[1].set_vort(i, j, omega0[j, i])
    t0 = time.perf_counter()
    num_steps = int(total_time / dt)
    for _ in range(num_steps):
        sim.step()
    mc_time = time.perf_counter() - t0
    mc_vort = sim.grid[sim.cur].vort.copy()
    mc_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"  MC 耗时: {mc_time:.2f}s")

    # ---------- 网格流函数法 ----------
    print("运行 Grid Stream Function...")
    grid_solver = GridStreamFunctionSolver(nx, ny, dt, dx, ox, oy, geo)
    grid_solver.set_omega(omega0)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        grid_solver.step()
    grid_time = time.perf_counter() - t0
    grid_vort = grid_solver.omega.copy()
    grid_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"  Grid 耗时: {grid_time:.2f}s")

    # ---------- 误差计算（域内） ----------
    def domain_error(v1, v2):
        mask = np.array([[geo.inside_domain(Vec2(ox+(i+0.5)*dx, oy+(j+0.5)*dx))
                          for i in range(nx)] for j in range(ny)])
        diff = (v1 - v2)[mask]
        ref = v2[mask]
        return float(np.linalg.norm(diff) / np.linalg.norm(ref))
    mc_l2 = domain_error(mc_vort, omega0)
    grid_l2 = domain_error(grid_vort, omega0)
    mc_grid_diff = domain_error(mc_vort, grid_vort)

    # ---------- 保存报告 ----------
    report_path = os.path.join(output_dir, "step2_mc_vs_grid_report.txt")
    with open(report_path, "w") as f:
        f.write("Step2: MC (WoS) vs Grid Stream Function 对比报告\n")
        f.write(f"网格 {nx}x{ny}, dt={dt}, T={total_time}s, 障碍物半径={args.obstacle_radius}\n\n")
        f.write(f"MC    L2误差(相对初始): {mc_l2:.6f}, 耗时: {mc_time:.2f}s, 内存: {mc_memory:.2f}MB\n")
        f.write(f"Grid  L2误差(相对初始): {grid_l2:.6f}, 耗时: {grid_time:.2f}s, 内存: {grid_memory:.2f}MB\n")
        f.write(f"MC与Grid差异: {mc_grid_diff:.6f}\n")
    print(f"\n报告已保存至 {report_path}")

    # ---------- 绘制对比图 ----------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    vmin, vmax = omega0.min(), omega0.max()
    axes[0].imshow(omega0, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Initial')
    axes[1].imshow(mc_vort, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'MC (nmc={args.nmc})')
    axes[2].imshow(grid_vort, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2].set_title('Grid')
    diff_plot = np.abs(mc_vort - grid_vort)
    axes[3].imshow(diff_plot, origin='lower', cmap='hot')
    axes[3].set_title('|MC - Grid|')
    for ax in axes:
        ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "step2_mc_vs_grid.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"对比图已保存至 {fig_path}")

if __name__ == '__main__':
    main()