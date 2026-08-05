import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.simulator import Simulator

def main():
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description='Step1 Taylor-Green 涡旋误差分析')
    parser.add_argument('--nx', type=int, default=64, help='网格横向分辨率')
    parser.add_argument('--ny', type=int, default=64, help='网格纵向分辨率')
    parser.add_argument('--dt', type=float, default=0.05, help='时间步长')
    parser.add_argument('--nmc', type=int, default=256, help='Monte Carlo 采样点数')
    parser.add_argument('--total_time', type=float, default=1.0, help='总模拟时长（秒）')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    args = parser.parse_args()

    nx, ny = args.nx, args.ny
    dt = args.dt
    nmc = args.nmc
    total_time = args.total_time
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ========== 创建模拟器 ==========
    sim = Simulator(nx=nx, ny=ny, dt=dt)
    sim.nmc = nmc

    # ========== 设置 Taylor-Green 初始涡量 ==========
    dx = sim.dx
    ox, oy = -1.0, -1.0
    for j in range(ny):
        for i in range(nx):
            x = ox + (i + 0.5) * dx
            y = oy + (j + 0.5) * dx
            w = 2.0 * np.cos(np.pi * x) * np.cos(np.pi * y)
            sim.grid[0].set_vort(i, j, w)
            sim.grid[1].set_vort(i, j, w)

    # 参考解（解析解，不随时间变化）
    omega_ref = sim.vorticity.copy()

    # 内部区域索引（去除边界影响，仅比较中心 80% 区域）
    margin = int(0.1 * nx)
    i_min, i_max = margin, nx - margin
    j_min, j_max = margin, ny - margin
    omega_ref_center = omega_ref[j_min:j_max, i_min:i_max]

    # ========== 记录误差 ==========
    times = [0.0]
    l2_errors = [0.0]
    linf_errors = [0.0]

    num_steps = int(total_time / dt)
    for step in range(1, num_steps + 1):
        sim.step()
        omega_mc = sim.vorticity
        omega_mc_center = omega_mc[j_min:j_max, i_min:i_max]

        diff = omega_mc_center - omega_ref_center
        l2 = float(np.linalg.norm(diff) / np.linalg.norm(omega_ref_center))
        linf = float(np.max(np.abs(diff)))
        t = step * dt

        times.append(t)
        l2_errors.append(l2)
        linf_errors.append(linf)
        print(f"t={t:.2f}s, L2误差={l2:.6f}, L∞误差={linf:.6f}")

    # ========== 保存误差报告 ==========
    report_path = os.path.join(output_dir, 'error_report.txt')
    with open(report_path, 'w') as f:
        f.write("Taylor-Green 涡旋误差分析 (2D 无粘 Euler)\n")
        f.write(f"网格: {nx}x{ny}, dt={dt}, nmc={nmc}, total_time={total_time}s\n")
        f.write(f"比较区域: 中心 80% (去除边界截断影响)\n\n")
        f.write("时间(s)\tL2相对误差\tL∞误差\n")
        for i, t in enumerate(times):
            f.write(f"{t:.2f}\t{l2_errors[i]:.6f}\t{linf_errors[i]:.6f}\n")
    print(f"\n误差报告已保存至 {report_path}")

    # ========== 绘制误差曲线 ==========
    plt.figure(figsize=(8, 5))
    plt.plot(times, l2_errors, marker='o', label='L2 relative error')
    plt.plot(times, linf_errors, marker='s', label='L∞ error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Taylor-Green Vortex: MC vs Exact Solution')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(output_dir, 'error_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"误差曲线已保存至 {curve_path}")

if __name__ == '__main__':
    main()