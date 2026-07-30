import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from core.simulator3d import Simulator3D

def save_animation_3d(sim: Simulator3D, total_time: float = 1.0,
                      output_dir: str = "output",
                      filename: str = "vorticity_slice.gif",
                      fps: int = 5, dpi: int = 80):
    """
    运行3D模拟，收集切片与全局误差指标，输出动画与误差报告。
    """
    os.makedirs(output_dir, exist_ok=True)

    num_steps = int(total_time / sim.dt)
    step_interval = max(1, int(1.0 / (fps * sim.dt)))

    # 存储数据
    slices = []
    times = []
    total_vort = []      # 总涡量幅值积分（L1范数）
    max_vort = []        # 最大涡量幅值

    # 初始帧
    slices.append(sim.vorticity_slice.copy())
    times.append(0.0)
    total_vort.append(np.sum(sim.vorticity_slice) * sim.dx**2 * sim.nz)  # 近似体积分（用切片近似，非全量）
    max_vort.append(np.max(sim.vorticity_slice))

    # 记录总耗时
    start_wall = time.perf_counter()
    step_times = []

    print(f"开始3D模拟，共 {num_steps} 步...")
    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()
        sim.step()
        step_time = time.perf_counter() - t0
        step_times.append(step_time)

        # 进度打印
        if step % max(1, num_steps // 10) == 0:
            print(f"  进度 {step/num_steps*100:.0f}% (步耗时 {step_time:.2f}s)")

        # 采集帧
        if step % step_interval == 0:
            sl = sim.vorticity_slice.copy()
            slices.append(sl)
            times.append(step * sim.dt)
            total_vort.append(np.sum(sl) * sim.dx**2 * sim.nz)
            max_vort.append(np.max(sl))

    total_wall = time.perf_counter() - start_wall
    avg_step_time = np.mean(step_times) if step_times else 0

    print(f"\n⏱ 模拟完成，总耗时: {total_wall:.2f}s")
    print(f"   平均每步耗时: {avg_step_time*1000:.0f}ms")

    # ---------- 误差分析 ----------
    print("\n--- 误差分析 ---")
    init_total = total_vort[0]
    final_total = total_vort[-1]
    total_change = (final_total - init_total) / init_total * 100
    print(f"  初始总涡量（切片积分）: {init_total:.6f}")
    print(f"  最终总涡量: {final_total:.6f}")
    print(f"  相对变化: {total_change:.2f}%")

    init_max = max_vort[0]
    final_max = max_vort[-1]
    max_change = (final_max - init_max) / init_max * 100
    print(f"  初始最大涡量: {init_max:.6f}")
    print(f"  最终最大涡量: {final_max:.6f}")
    print(f"  相对变化: {max_change:.2f}%")

    # 写入误差报告
    report_path = os.path.join(output_dir, "error_report.txt")
    with open(report_path, "w") as f:
        f.write(f"模拟参数: nx={sim.nx}, ny={sim.ny}, nz={sim.nz}\n")
        f.write(f"dx={sim.dx:.6f}, dt={sim.dt}, nu={sim.nu}\n")
        f.write(f"nmc={sim.nmc}, nd={sim.nd}\n")
        f.write(f"总模拟时间: {total_time}s\n")
        f.write(f"总耗时: {total_wall:.2f}s, 平均每步: {avg_step_time*1000:.0f}ms\n\n")
        f.write("时间(s)  切片总涡量  最大涡量\n")
        for t, tv, mv in zip(times, total_vort, max_vort):
            f.write(f"{t:.2f}    {tv:.6f}  {mv:.6f}\n")
    print(f"   误差报告已保存至: {report_path}")

    # 保存最后一帧为图片（静态）
    last_slice_path = os.path.join(output_dir, "last_frame.png")
    plt.imsave(last_slice_path, slices[-1], origin='lower', cmap='hot', vmin=0, vmax=0.6)
    print(f"   最后一帧已保存至: {last_slice_path}")

    # 生成动画
    print("正在生成动画...")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(slices[0], origin='lower', cmap='hot', vmin=0, vmax=0.6)
    ax.set_title('Vorticity magnitude slice z=0')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        color='w', fontsize=12, va='top')

    def update(idx):
        im.set_array(slices[idx])
        time_text.set_text(f't={times[idx]:.2f}s')
        return [im, time_text]

    anim = FuncAnimation(fig, update, frames=len(slices),
                         interval=1000 // fps, blit=True)
    anim_path = os.path.join(output_dir, filename)
    anim.save(anim_path, writer=PillowWriter(fps=fps), dpi=dpi)
    print(f"✅ 动画已保存至: {anim_path}")
    plt.close(fig)