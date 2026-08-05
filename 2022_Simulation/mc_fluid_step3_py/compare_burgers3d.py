import sys, os, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.simulator3d import Simulator3D
from core.types3d import Vec3

# ============================================================
# Burgers 涡解析涡量场（稳态）
# ============================================================
def burgers_vorticity(x, y, z, a, Gamma, nu):
    """
    返回 Burgers 涡的解析涡量 (omega_x, omega_y, omega_z)。
    omega_x = omega_y = 0, omega_z = (a*Gamma)/(2*pi*nu) * exp(-a*r^2/(2*nu))
    其中 r^2 = x^2 + y^2。
    """
    r2 = x*x + y*y
    prefactor = a * Gamma / (2.0 * np.pi * nu)
    wz = prefactor * np.exp(-a * r2 / (2.0 * nu))
    return 0.0, 0.0, wz

# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Step3: Burgers 涡误差分析 + 动画')
    parser.add_argument('--nx', type=int, default=32, help='网格 x 分辨率')
    parser.add_argument('--ny', type=int, default=32)
    parser.add_argument('--nz', type=int, default=32)
    parser.add_argument('--dt', type=float, default=0.02, help='时间步长')
    parser.add_argument('--total_time', type=float, default=1.0, help='总模拟时长（秒）')
    parser.add_argument('--nmc', type=int, default=64, help='MC 速度采样数')
    parser.add_argument('--nu', type=float, default=0.05, help='粘性系数')
    parser.add_argument('--a', type=float, default=1.0, help='拉伸参数 a')
    parser.add_argument('--Gamma', type=float, default=1.0, help='环量 Gamma')
    parser.add_argument('--L', type=float, default=1.0, help='域半边长（域为[-L,L]^3）')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--fps', type=int, default=10, help='动画帧率')
    args = parser.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    dt, total_time = args.dt, args.total_time
    L = args.L
    a, Gamma, nu = args.a, args.Gamma, args.nu

    dx = (2.0 * L) / nx
    ox = oy = oz = -L

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 创建模拟器 ----------
    sim = Simulator3D(nx=nx, ny=ny, nz=nz, dt=dt, nu=nu,)
    sim.nmc = args.nmc

    # 初始化涡量为 Burgers 涡解析场
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = ox + (i + 0.5) * dx
                y = oy + (j + 0.5) * dx
                z = oz + (k + 0.5) * dx
                wx, wy, wz = burgers_vorticity(x, y, z, a, Gamma, nu)
                sim.grid[0].set_vort(i, j, k, Vec3(wx, wy, wz))
                sim.grid[1].set_vort(i, j, k, Vec3(wx, wy, wz))

    # ---------- 初始化记录变量 ----------
    num_steps = int(total_time / dt)
    times = [0.0]
    l2_errors = [0.0]
    linf_errors = [0.0]

    # 动画帧记录（z=0 切片幅值，等于 |omega_z| 因为其他分量为零）
    frame_times = []
    sim_frames = []
    ref_frames = []

    # 中心区域索引（内部 60%）
    margin = int(0.2 * nx)
    record_interval = max(1, num_steps // 30)
    k0 = nz // 2

    print(f"Burgers 涡误差分析 ({nx}³, L={L}, a={a}, Γ={Gamma}, ν={nu}, nmc={args.nmc})...")

    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()
        sim.step()
        elapsed = time.perf_counter() - t0
        t = step * dt

        vort_sim = sim.grid[sim.cur].vort   # (nz, ny, nx, 3)

        # 计算中心区域误差（只比较涡量分量，因为解析解只有ωz非零）
        diff_sq_sum = 0.0
        ref_sq_sum = 0.0
        max_diff = 0.0
        for k in range(margin, nx - margin):
            for j in range(margin, ny - margin):
                for i in range(margin, nx - margin):
                    x = ox + (i + 0.5) * dx
                    y = oy + (j + 0.5) * dx
                    z = oz + (k + 0.5) * dx
                    wx_ref, wy_ref, wz_ref = burgers_vorticity(x, y, z, a, Gamma, nu)
                    # 由于模拟中的涡量可能有三个分量，这里计算整个向量的误差
                    wx_sim, wy_sim, wz_sim = vort_sim[k, j, i]
                    diff = (wx_sim - wx_ref)**2 + (wy_sim - wy_ref)**2 + (wz_sim - wz_ref)**2
                    ref = wx_ref**2 + wy_ref**2 + wz_ref**2
                    diff_sq_sum += diff
                    ref_sq_sum += ref
                    max_diff = max(max_diff, np.sqrt(diff))

        l2 = np.sqrt(diff_sq_sum / ref_sq_sum) if ref_sq_sum > 0 else 0.0
        linf = max_diff
        times.append(t)
        l2_errors.append(l2)
        linf_errors.append(linf)

        # 记录动画帧（z=0 切片，显示|ω|）
        if step % record_interval == 0 or step == num_steps:
            sim_slice = np.zeros((ny, nx))
            ref_slice = np.zeros((ny, nx))
            for j in range(ny):
                for i in range(nx):
                    x = ox + (i + 0.5) * dx
                    y = oy + (j + 0.5) * dx
                    z = oz + (k0 + 0.5) * dx
                    wx_s, wy_s, wz_s = vort_sim[k0, j, i]
                    sim_slice[j, i] = np.sqrt(wx_s**2 + wy_s**2 + wz_s**2)
                    wx_r, wy_r, wz_r = burgers_vorticity(x, y, z, a, Gamma, nu)
                    ref_slice[j, i] = np.sqrt(wx_r**2 + wy_r**2 + wz_r**2)
            sim_frames.append(sim_slice)
            ref_frames.append(ref_slice)
            frame_times.append(t)

        if step % max(1, num_steps // 10) == 0:
            print(f"  Step {step}/{num_steps} (t={t:.2f}s, 耗时 {elapsed:.2f}s): L2={l2:.6f}, L∞={linf:.6f}")

    # ---------- 保存报告 ----------
    report_path = os.path.join(output_dir, "step3_burgers_error_report.txt")
    with open(report_path, "w") as f:
        f.write("Step3: Burgers 涡误差分析\n")
        f.write(f"网格 {nx}³, L={L}, dt={dt}, total_time={total_time}s, a={a}, Γ={Gamma}, ν={nu}, nmc={args.nmc}\n")
        f.write(f"比较区域: 中心 60% (去除边界)\n\n")
        f.write("时间(s)\tL2相对误差\tL∞误差\n")
        for i, t in enumerate(times):
            f.write(f"{t:.2f}\t{l2_errors[i]:.6f}\t{linf_errors[i]:.6f}\n")
    print(f"\n报告已保存至 {report_path}")

    # ---------- 误差曲线 ----------
    plt.figure(figsize=(8,5))
    plt.plot(times, l2_errors, 'o-', label='L2 relative error')
    plt.plot(times, linf_errors, 's-', label='L∞ error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title(f'Burgers Vortex (a={a}, Γ={Gamma}, ν={nu}) - L2 and L∞ Errors')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(output_dir, "step3_burgers_error_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"误差曲线已保存至 {curve_path}")

    # ---------- 最终切片对比（z=0） ----------
    print("正在绘制最终涡量切片对比图...")
    vort_sim_final = sim.grid[sim.cur].vort
    sim_slice_mag = np.zeros((ny, nx))
    ref_slice_mag = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            x = ox + (i + 0.5) * dx
            y = oy + (j + 0.5) * dx
            z = oz + (k0 + 0.5) * dx
            wx_s, wy_s, wz_s = vort_sim_final[k0, j, i]
            sim_slice_mag[j, i] = np.sqrt(wx_s**2 + wy_s**2 + wz_s**2)
            wx_r, wy_r, wz_r = burgers_vorticity(x, y, z, a, Gamma, nu)
            ref_slice_mag[j, i] = np.sqrt(wx_r**2 + wy_r**2 + wz_r**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmin = min(sim_slice_mag.min(), ref_slice_mag.min())
    vmax = max(sim_slice_mag.max(), ref_slice_mag.max())
    im1 = axes[0].imshow(sim_slice_mag, origin='lower', cmap='hot',
                         vmin=vmin, vmax=vmax,
                         extent=[ox, ox+nx*dx, oy, oy+ny*dx])
    axes[0].set_title('MC Simulation (t = {:.2f} s)'.format(total_time))
    plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(ref_slice_mag, origin='lower', cmap='hot',
                         vmin=vmin, vmax=vmax,
                         extent=[ox, ox+nx*dx, oy, oy+ny*dx])
    axes[1].set_title('Analytical (t = {:.2f} s)'.format(total_time))
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    slice_path = os.path.join(output_dir, "step3_burgers_final_slice.png")
    plt.savefig(slice_path, dpi=150)
    plt.close()
    print(f"最终切片图已保存至 {slice_path}")

    # ---------- MC vs Analytical 动画 ----------
    print("正在生成 MC vs Analytical 动画...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmin_anim = min(np.min(f) for f in sim_frames + ref_frames)
    vmax_anim = max(np.max(f) for f in sim_frames + ref_frames)
    im_sim = axes[0].imshow(sim_frames[0], origin='lower', cmap='hot',
                            vmin=vmin_anim, vmax=vmax_anim,
                            extent=[ox, ox+nx*dx, oy, oy+ny*dx])
    axes[0].set_title('MC Simulation')
    plt.colorbar(im_sim, ax=axes[0])
    im_ref = axes[1].imshow(ref_frames[0], origin='lower', cmap='hot',
                            vmin=vmin_anim, vmax=vmax_anim,
                            extent=[ox, ox+nx*dx, oy, oy+ny*dx])
    axes[1].set_title('Analytical')
    plt.colorbar(im_ref, ax=axes[1])
    time_text = fig.suptitle('t = {:.2f} s'.format(frame_times[0]))

    def update(idx):
        im_sim.set_array(sim_frames[idx])
        im_ref.set_array(ref_frames[idx])
        time_text.set_text('t = {:.2f} s'.format(frame_times[idx]))
        return [im_sim, im_ref, time_text]

    anim = FuncAnimation(fig, update, frames=len(sim_frames),
                         interval=1000 // args.fps, blit=False)
    anim_path = os.path.join(output_dir, "step3_burgers_evolution.gif")
    anim.save(anim_path, writer=PillowWriter(fps=args.fps), dpi=100)
    plt.close()
    print(f"对比动画已保存至 {anim_path}")

    # ---------- 理论演化动画（仅解析解） ----------
    print("正在生成理论演化动画...")
    fig, ax = plt.subplots(figsize=(5, 5))
    vmin_theory = np.min(ref_frames)
    vmax_theory = np.max(ref_frames)
    im_theory = ax.imshow(ref_frames[0], origin='lower', cmap='hot',
                          vmin=vmin_theory, vmax=vmax_theory,
                          extent=(ox, ox+nx*dx, oy, oy+ny*dx))
    ax.set_title('Analytical Burgers Vortex Evolution')
    plt.colorbar(im_theory, ax=ax)
    time_text_theory = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                               color='white', fontsize=12, va='top')

    def update_theory(idx):
        im_theory.set_array(ref_frames[idx])
        time_text_theory.set_text('t = {:.2f} s'.format(frame_times[idx]))
        return [im_theory, time_text_theory]

    anim_theory = FuncAnimation(fig, update_theory, frames=len(ref_frames),
                                interval=1000 // args.fps, blit=False)
    theory_path = os.path.join(output_dir, "step3_burgers_analytical_evolution.gif")
    anim_theory.save(theory_path, writer=PillowWriter(fps=args.fps), dpi=100)
    plt.close()
    print(f"理论演化动画已保存至 {theory_path}")

if __name__ == '__main__':
    main()