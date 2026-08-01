import os, time, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_and_report(sim, total_time, output_dir, filename,
                       fps=5, dpi=80, label=''):
    """通用动画与误差报告函数（可被主程序调用）"""
    os.makedirs(output_dir, exist_ok=True)
    num_steps = int(total_time / sim.dt)
    step_interval = max(1, int(1.0 / (fps * sim.dt)))
    slices, times = [], []
    total_vort, max_vort = [], []

    slices.append(sim.vorticity_slice.copy()); times.append(0.0)
    total_vort.append(np.sum(sim.vorticity_slice) * sim.dx**2)
    max_vort.append(np.max(sim.vorticity_slice))

    step_times = []
    for step in range(1, num_steps+1):
        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter()-t0)
        if step % step_interval == 0:
            sl = sim.vorticity_slice.copy()
            slices.append(sl); times.append(step*sim.dt)
            total_vort.append(np.sum(sl) * sim.dx**2)
            max_vort.append(np.max(sl))

    # 保存报告
    report_path = os.path.join(output_dir, f"report{label}.txt")
    with open(report_path, 'w') as f:
        f.write(f'Simulation: {label}\n')
        f.write(f'nx={sim.nx}, dt={sim.dt}, nu={sim.nu}, nmc={sim.nmc}\n')
        f.write(f'Total time: {total_time}s\n')
        f.write(f'Wall time: {sum(step_times):.2f}s, avg step: {np.mean(step_times)*1000:.0f}ms\n\n')
        f.write('time(s)\ttotal\tmax\n')
        for t,tv,mv in zip(times,total_vort,max_vort):
            f.write(f'{t:.2f}\t{tv:.6f}\t{mv:.6f}\n')

    # 保存动画
    anim_path = os.path.join(output_dir, f"anim{label}.gif")
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(slices[0], origin='lower', cmap='hot', vmin=0, vmax=0.8)
    ax.set_title(f'Vorticity slice z=0 ({label})')
    time_text = ax.text(0.02,0.95,'', transform=ax.transAxes, color='w', fontsize=12)
    def update(idx):
        im.set_array(slices[idx])
        time_text.set_text(f't={times[idx]:.2f}s')
        return [im,time_text]
    anim = FuncAnimation(fig, update, frames=len(slices), interval=1000//fps, blit=True)
    anim.save(anim_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f'[OK] {label}: anim saved to {anim_path}, report saved to {report_path}')