# render/visualizer.py
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from core.simulator import Simulator


def save_animation(sim: Simulator, total_time: float = 10.0,
                   output_dir: str = "graph", filename: str = "vorticity_evolution.gif",
                   fps: int = 10, dpi: int = 80,
                   calc_error: bool = True):
    """
    运行模拟，保存动画，并输出耗时与误差分析。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------- 开始计时 ----------
    start_time = time.perf_counter()

    num_steps = int(total_time / sim.dt)
    step_interval = max(1, int(1.0 / (fps * sim.dt)))

    frames = []
    times = []
    circulation = []   # 存储每帧的总环量（用于误差分析）
    max_vort = []      # 存储每帧的最大涡量

    # 初始帧
    frames.append(sim.vorticity.copy())
    times.append(0.0)
    if calc_error:
        dx2 = sim.grid[0].dx * sim.grid[0].dx
        circulation.append(np.sum(sim.vorticity) * dx2)
        max_vort.append(np.max(np.abs(sim.vorticity)))

    print(f"开始模拟，共 {num_steps} 步...")
    for step in range(1, num_steps + 1):
        sim.step()
        # 进度打印
        if step % max(1, num_steps // 10) == 0:
            print(f"  进度: {step/num_steps*100:.0f}%")
        if step % step_interval == 0:
            frames.append(sim.vorticity.copy())
            cur_time = step * sim.dt
            times.append(cur_time)
            if calc_error:
                dx2 = sim.grid[0].dx * sim.grid[0].dx
                circulation.append(np.sum(sim.vorticity) * dx2)
                max_vort.append(np.max(np.abs(sim.vorticity)))

    # 模拟耗时
    sim_time = time.perf_counter() - start_time
    avg_step_time = sim_time / num_steps

    print(f"\n⏱ 模拟完成，总耗时: {sim_time:.2f} 秒")
    print(f"   平均每步耗时: {avg_step_time*1000:.1f} 毫秒")

    # ---------- 误差分析 ----------
    if calc_error:
        print("\n--- 误差分析 ---")
        # 环量守恒：理论应与初始 t=0 相同
        circ0 = circulation[0]
        circ_final = circulation[-1]
        circ_change = (circ_final - circ0) / circ0 * 100
        print(f"  初始总环量: {circ0:.6f}")
        print(f"  最终总环量: {circ_final:.6f}")
        print(f"  环量相对变化: {circ_change:.2f}%")
        # 最大涡量衰减
        max0 = max_vort[0]
        max_final = max_vort[-1]
        max_decay = (max_final - max0) / max0 * 100
        print(f"  初始最大涡量: {max0:.6f}")
        print(f"  最终最大涡量: {max_final:.6f}")
        print(f"  最大涡量变化: {max_decay:.2f}%")

        # 将误差数据保存到文本文件
        error_log = os.path.join(output_dir, "error_report.txt")
        with open(error_log, "w") as f:
            f.write(f"模拟参数: nx={sim.nx}, dt={sim.dt}, nmc={sim.nmc}\n")
            f.write(f"总模拟时间: {total_time}s\n")
            f.write(f"总耗时: {sim_time:.2f}s, 平均每步: {avg_step_time*1000:.1f}ms\n\n")
            f.write("时间(s)  总环量    最大涡量\n")
            for t, circ, mv in zip(times, circulation, max_vort):
                f.write(f"{t:.2f}    {circ:.6f}  {mv:.6f}\n")
        print(f"   详细误差报告已保存至: {error_log}")

    # 保存最后一帧测试
    test_path = os.path.join(output_dir, "last_frame.png")
    plt.imsave(test_path, frames[-1], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    print(f"   最后一帧已保存至: {test_path}")

    # ---------- 生成动画 ----------
    print("正在生成动画...")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0], origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, extent=(-1, 1, -1, 1))
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        color='white', fontsize=12, va='top')
    ax.set_title('MC Fluid – Vorticity Evolution')

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        time_text.set_text(f't = {times[frame_idx]:.2f}s')
        return [im, time_text]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=True)
    output_path = os.path.join(output_dir, filename)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    print(f"✅ 动画已保存至: {output_path}")
    plt.close(fig)