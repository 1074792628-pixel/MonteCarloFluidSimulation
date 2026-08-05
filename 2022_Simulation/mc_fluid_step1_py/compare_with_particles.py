import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.simulator import Simulator
from core.vortex_solver import VortexParticleSolver

# ============================================================
# 初始化 Taylor-Green 涡量场
# ============================================================
def taylor_green_grid(nx, ny, dx, ox, oy):
    grid = np.zeros((ny, nx))
    for j in range(ny):
        y = oy + (j + 0.5) * dx
        for i in range(nx):
            x = ox + (i + 0.5) * dx
            grid[j, i] = 2.0 * np.cos(np.pi * x) * np.cos(np.pi * y)
    return grid

# ============================================================
# 运行 MC 模拟（一次）
# ============================================================
def run_mc(nx, ny, dt, total_time, nmc, omega0):
    sim = Simulator(nx=nx, ny=ny, dt=dt)
    sim.nmc = nmc
    for j in range(ny):
        for i in range(nx):
            w = omega0[j, i]
            sim.grid[0].set_vort(i, j, w)
            sim.grid[1].set_vort(i, j, w)

    num_steps = int(total_time / dt)
    sim_time = 0.0
    for _ in range(num_steps):
        t0 = time.perf_counter()
        sim.step()
        sim_time += time.perf_counter() - t0
    return sim.vorticity, sim_time

# ============================================================
# 运行粒子法（一次）——已修复 dx 参数
# ============================================================
def run_particle(nx, ny, dt, total_time, omega0, ox, oy, dx):
    solver = VortexParticleSolver(nx, ny, dt, dx)
    solver.initialize_from_grid(omega0, ox, oy)

    num_steps = int(total_time / dt)
    sim_time = 0.0
    for _ in range(num_steps):
        t0 = time.perf_counter()
        solver.step()
        sim_time += time.perf_counter() - t0
    vort_grid = solver.get_vorticity_grid(nx, ny, ox, oy)
    return vort_grid, sim_time

# ============================================================
# 误差计算（中心 80% 区域）
# ============================================================
def compute_l2_error(mc_grid, ref_grid, nx, ny):
    margin = int(0.1 * nx)
    mc_c = mc_grid[margin:ny-margin, margin:nx-margin]
    ref_c = ref_grid[margin:ny-margin, margin:nx-margin]
    return float(np.linalg.norm(mc_c - ref_c) / np.linalg.norm(ref_c))

# ============================================================
# 主程序
# ============================================================
def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='MC vs Vortex Particle 对比实验（Taylor-Green）')
    parser.add_argument('--nx', type=int, default=32, help='网格横向分辨率')
    parser.add_argument('--ny', type=int, default=32, help='网格纵向分辨率')
    parser.add_argument('--dt', type=float, default=0.02, help='时间步长')
    parser.add_argument('--total_time', type=float, default=0.5, help='总模拟时长（秒）')
    parser.add_argument('--nmc_list', type=str, default='16,64,256,1024', help='MC样本数列表（逗号分隔）')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    args = parser.parse_args()

    # 参数
    nx, ny = args.nx, args.ny
    dt = args.dt
    total_time = args.total_time
    ox, oy = -1.0, -1.0
    dx = 2.0 / nx
    mc_configs = [int(x) for x in args.nmc_list.split(',')]

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 初始场（参考解，无粘稳态）
    omega0 = taylor_green_grid(nx, ny, dx, ox, oy)
    ref_grid = omega0.copy()

    # ---------------- MC 方法 ----------------
    mc_results = []
    for nmc in mc_configs:
        print(f"运行 MC nmc={nmc} ...")
        vort, sim_time = run_mc(nx, ny, dt, total_time, nmc, omega0)
        l2 = compute_l2_error(vort, ref_grid, nx, ny)
        mc_results.append((sim_time, l2, nmc))
        print(f"  MC nmc={nmc}: L2={l2:.6f}, 时间={sim_time:.2f}s")

    # ---------------- 粒子法 ----------------
    print("运行 涡量粒子法 ...")
    particle_vort, particle_time = run_particle(nx, ny, dt, total_time, omega0, ox, oy, dx)
    particle_l2 = compute_l2_error(particle_vort, ref_grid, nx, ny)
    particle_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"  粒子法: L2={particle_l2:.6f}, 时间={particle_time:.2f}s, 内存={particle_memory:.2f}MB")

    # MC 内存（简单采样）
    mc_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2

    # ============================================================
    # 保存报告
    # ============================================================
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("MC vs Vortex Particle 对比报告\n")
        f.write(f"基准: Taylor-Green 涡旋, 网格 {nx}x{ny}, dt={dt}, T={total_time}s\n\n")
        f.write("方法\tL2误差\t计算时间(s)\t内存(MB)\n")
        for sim_time, l2, nmc in mc_results:
            f.write(f"MC(nmc={nmc})\t{l2:.6f}\t{sim_time:.2f}\t{mc_memory:.2f}\n")
        f.write(f"VortexParticle\t{particle_l2:.6f}\t{particle_time:.2f}\t{particle_memory:.2f}\n")
    print(f"\n报告已保存至 {report_path}")

    # ============================================================
    # 图1：误差随时间演化（固定 MC nmc=256）
    # ============================================================
    fixed_nmc = 256 if 256 in mc_configs else mc_configs[-1]
    mc_times_evo, mc_errors_evo = [], []
    particle_times_evo, particle_errors_evo = [], []

    sim = Simulator(nx=nx, ny=ny, dt=dt)
    sim.nmc = fixed_nmc
    for j in range(ny):
        for i in range(nx):
            sim.grid[0].set_vort(i, j, omega0[j, i])
            sim.grid[1].set_vort(i, j, omega0[j, i])

    solver = VortexParticleSolver(nx, ny, dt, dx)
    solver.initialize_from_grid(omega0, ox, oy)

    num_steps = int(total_time / dt)
    for step in range(1, num_steps + 1):
        sim.step()
        solver.step()
        t = step * dt
        if step % max(1, num_steps // 20) == 0:
            mc_vort = sim.vorticity
            par_vort = solver.get_vorticity_grid(nx, ny, ox, oy)
            mc_times_evo.append(t)
            mc_errors_evo.append(compute_l2_error(mc_vort, ref_grid, nx, ny))
            particle_times_evo.append(t)
            particle_errors_evo.append(compute_l2_error(par_vort, ref_grid, nx, ny))

    plt.figure(figsize=(8, 5))
    plt.plot(mc_times_evo, mc_errors_evo, 'o-', label=f'MC (nmc={fixed_nmc})')
    plt.plot(particle_times_evo, particle_errors_evo, 's-', label='Vortex Particle')
    plt.xlabel('Time (s)')
    plt.ylabel('L2 Relative Error')
    plt.title('L2 Error Evolution: MC vs Vortex Particle')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_vs_time.png"), dpi=150)
    plt.close()
    print(f"误差-时间曲线已保存至 {os.path.join(output_dir, 'error_vs_time.png')}")

    # ============================================================
    # 图2：Pareto 曲线（误差 vs 计算时间）
    # ============================================================
    pareto_times = [t for t, _, _ in mc_results] + [particle_time]
    pareto_errors = [l2 for _, l2, _ in mc_results] + [particle_l2]
    pareto_labels = [f'MC n={n}' for _, _, n in mc_results] + ['Vortex Particle']

    plt.figure(figsize=(8, 5))
    plt.loglog(pareto_times, pareto_errors, 'o-', markersize=8)
    for x, y, label in zip(pareto_times, pareto_errors, pareto_labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(8, -8), fontsize=9)
    plt.xlabel('Computation Time (s)')
    plt.ylabel('L2 Relative Error')
    plt.title('Error vs. Computation Time (Pareto)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "pareto_curve.png"), dpi=150)
    plt.close()
    print(f"Pareto 曲线已保存至 {os.path.join(output_dir, 'pareto_curve.png')}")

    print(f"\n内存使用: MC≈{mc_memory:.2f}MB, 粒子法≈{particle_memory:.2f}MB")

if __name__ == '__main__':
    main()