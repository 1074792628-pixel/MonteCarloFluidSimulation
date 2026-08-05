import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保脚本所在目录的父目录在 PATH 中（以便导入 core/）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.simulator import Simulator

def run_once(nx, ny, dt, nmc, total_time):
    """运行一次模拟，返回 t=total_time 时的 L2 误差"""
    sim = Simulator(nx=nx, ny=ny, dt=dt)
    sim.nmc = nmc

    # 设置 Taylor-Green 初始涡量
    dx = sim.dx
    ox, oy = -1.0, -1.0
    for j in range(ny):
        for i in range(nx):
            x = ox + (i + 0.5) * dx
            y = oy + (j + 0.5) * dx
            w = 2.0 * np.cos(np.pi * x) * np.cos(np.pi * y)
            sim.grid[0].set_vort(i, j, w)
            sim.grid[1].set_vort(i, j, w)

    omega_ref = sim.vorticity.copy()
    margin = int(0.1 * nx)
    omega_ref_center = omega_ref[margin:ny-margin, margin:nx-margin]

    num_steps = int(total_time / dt)
    for _ in range(num_steps):
        sim.step()

    omega_mc = sim.vorticity
    omega_mc_center = omega_mc[margin:ny-margin, margin:nx-margin]
    diff = omega_mc_center - omega_ref_center
    l2 = float(np.linalg.norm(diff) / np.linalg.norm(omega_ref_center))
    return l2

def main():
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description='Step1 Taylor-Green 误差随样本数收敛性分析')
    parser.add_argument('--nx', type=int, default=64, help='网格横向分辨率（默认 64）')
    parser.add_argument('--ny', type=int, default=64, help='网格纵向分辨率（默认 64）')
    parser.add_argument('--dt', type=float, default=0.02, help='时间步长（默认 0.02）')
    parser.add_argument('--total_time', type=float, default=1.0, help='总模拟时长（默认 1.0s）')
    parser.add_argument('--nmc_list', type=str, default='16,32,64,128,256,512',
                        help='MC样本数列表，逗号分隔（默认 "16,32,64,128,256,512"）')
    parser.add_argument('--repeats', type=int, default=5, help='重复次数（默认 5）')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录（默认 ./output）')
    args = parser.parse_args()

    # 解析参数
    nx, ny = args.nx, args.ny
    dt = args.dt
    total_time = args.total_time
    nmc_list = [int(x) for x in args.nmc_list.split(',')]
    repeats = args.repeats
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)  # 强制保存到脚本所在目录下的 output 文件夹
    os.makedirs(output_dir, exist_ok=True)

    # ========== 开始研究 ==========
    mean_errors = []
    std_errors = []

    print("研究误差随样本数 n 的变化")
    print(f"网格 {nx}x{ny}, dt={dt}, 总时间 {total_time}s, 重复 {repeats} 次\n")
    print("nmc\t平均L2误差\t标准差")

    for nmc in nmc_list:
        errors = []
        for _ in range(repeats):
            errors.append(run_once(nx, ny, dt, nmc, total_time))
        mean = float(np.mean(errors))
        std = float(np.std(errors))
        mean_errors.append(mean)
        std_errors.append(std)
        print(f"{nmc}\t{mean:.6f}\t{std:.6f}")

    # ========== 保存 txt 文件 ==========
    txt_path = os.path.join(output_dir, "nmc_study.txt")
    with open(txt_path, "w") as f:
        f.write("nmc\tmean_L2\tstd_L2\n")
        for nmc, m, s in zip(nmc_list, mean_errors, std_errors):
            f.write(f"{nmc}\t{m:.6f}\t{s:.6f}\n")
    print(f"\n数据已保存至 {txt_path}")

    # ========== 绘制 log-log 图 ==========
    plt.figure(figsize=(7,5))
    plt.loglog(nmc_list, mean_errors, 'o-', label='mean L2 error')
    # 拟合参考线 1/sqrt(n)（以第一个点为起点）
    if len(nmc_list) > 0 and mean_errors[0] > 0:
        x_fit = np.array(nmc_list)
        y_fit = mean_errors[0] * np.sqrt(nmc_list[0]) / np.sqrt(x_fit)
        plt.loglog(x_fit, y_fit, '--', label='O(1/√n) reference')
    plt.xlabel('MC samples n')
    plt.ylabel('L2 relative error')
    plt.title('Error vs. MC Sample Count')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    fig_path = os.path.join(output_dir, "nmc_error_convergence.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"收敛曲线已保存至 {fig_path}")

if __name__ == '__main__':
    main()