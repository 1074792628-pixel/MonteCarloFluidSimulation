import sys, os, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.simulator3d import Simulator3D
from core.types3d import Vec3

# ============================================================
# 3D Taylor-Green 解析涡量场
# ============================================================
def tg_vorticity(x, y, z, t, nu):
    decay = np.exp(-nu * np.pi**2 * t)
    wx = -np.pi * np.cos(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z) * decay
    wy =  np.pi * np.sin(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z) * decay
    wz =  2.0*np.pi * np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(np.pi*z) * decay
    return wx, wy, wz


# ============================================================
# 运行一次模拟，返回 t=total_time 时中心区域的 L2 误差
# ============================================================
def run_once(nx, ny, nz, dt, total_time, nmc, nu):
    sim = Simulator3D(nx=nx, ny=ny, nz=nz, dt=dt, nu=nu)
    sim.nmc = nmc

    dx = 2.0 / nx
    ox = oy = oz = -1.0

    # 设置初始涡量为 TG t=0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = ox + (i+0.5)*dx
                y = oy + (j+0.5)*dx
                z = oz + (k+0.5)*dx
                wx, wy, wz = tg_vorticity(x, y, z, 0.0, nu)
                sim.grid[0].set_vort(i, j, k, Vec3(wx, wy, wz))
                sim.grid[1].set_vort(i, j, k, Vec3(wx, wy, wz))

    num_steps = int(total_time / dt)
    for _ in range(num_steps):
        sim.step()

    # 当前涡量
    vort_sim = sim.grid[sim.cur].vort   # (nz, ny, nx, 3)

    # 中心 60% 区域
    margin = int(0.2 * nx)
    i_s = slice(margin, nx-margin)
    j_s = slice(margin, ny-margin)
    k_s = slice(margin, nz-margin)

    diff_sq = 0.0
    ref_sq = 0.0
    for k in range(margin, nz-margin):
        for j in range(margin, ny-margin):
            for i in range(margin, nx-margin):
                x = ox + (i+0.5)*dx
                y = oy + (j+0.5)*dx
                z = oz + (k+0.5)*dx
                wx_ref, wy_ref, wz_ref = tg_vorticity(x, y, z, total_time, nu)
                wx_sim, wy_sim, wz_sim = vort_sim[k, j, i]
                diff_sq += (wx_sim-wx_ref)**2 + (wy_sim-wy_ref)**2 + (wz_sim-wz_ref)**2
                ref_sq  += wx_ref**2 + wy_ref**2 + wz_ref**2

    l2 = np.sqrt(diff_sq / ref_sq) if ref_sq > 0 else 0.0
    return float(l2)


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Step3: MC 误差随样本数变化研究')
    parser.add_argument('--nx', type=int, default=16, help='网格 x/y/z 分辨率')
    parser.add_argument('--dt', type=float, default=0.02, help='时间步长')
    parser.add_argument('--total_time', type=float, default=0.5, help='总模拟时长')
    parser.add_argument('--nu', type=float, default=0.01, help='粘性系数')
    parser.add_argument('--nmc_list', type=str, default='16,32,64,128,256',
                        help='MC 样本数列表（逗号分隔）')
    parser.add_argument('--repeats', type=int, default=3, help='重复次数')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    args = parser.parse_args()

    nx = ny = nz = args.nx
    dt, total_time, nu = args.dt, args.total_time, args.nu
    nmc_list = [int(x) for x in args.nmc_list.split(',')]
    repeats = args.repeats
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mean_errors = []
    std_errors = []

    print(f"Step3 MC 误差随样本数研究")
    print(f"网格 {nx}³, dt={dt}, total_time={total_time}s, ν={nu}, repeats={repeats}\n")
    print("nmc\t平均L2误差\t标准差")

    for nmc in nmc_list:
        errors = []
        for _ in range(repeats):
            errors.append(run_once(nx, ny, nz, dt, total_time, nmc, nu))
        mean = float(np.mean(errors))
        std = float(np.std(errors))
        mean_errors.append(mean)
        std_errors.append(std)
        print(f"{nmc}\t{mean:.6f}\t{std:.6f}")

    # 保存 txt
    txt_path = os.path.join(output_dir, "step3_nmc_study.txt")
    with open(txt_path, "w") as f:
        f.write("Step3 MC 误差随样本数研究\n")
        f.write(f"网格 {nx}³, dt={dt}, total_time={total_time}s, ν={nu}\n\n")
        f.write("nmc\tmean_L2\tstd_L2\n")
        for nmc, m, s in zip(nmc_list, mean_errors, std_errors):
            f.write(f"{nmc}\t{m:.6f}\t{s:.6f}\n")
    print(f"\n数据已保存至 {txt_path}")

    # 绘制 log-log 图
    plt.figure(figsize=(7,5))
    plt.loglog(nmc_list, mean_errors, 'o-', label='mean L2 error')
    if mean_errors[0] > 0:
        x_fit = np.array(nmc_list, dtype=float)
        y_fit = mean_errors[0] * np.sqrt(nmc_list[0]) / np.sqrt(x_fit)
        plt.loglog(x_fit, y_fit, '--', label='O(1/√n) reference')
    plt.xlabel('MC samples n')
    plt.ylabel('L2 relative error')
    plt.title('Step3: Error vs. MC Sample Count')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    fig_path = os.path.join(output_dir, "step3_nmc_error_convergence.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"收敛曲线已保存至 {fig_path}")

if __name__ == '__main__':
    main()