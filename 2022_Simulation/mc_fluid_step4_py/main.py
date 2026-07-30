import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulator4 import Simulator4
from render.visualizer4 import animate_and_report

def main():
    parser = argparse.ArgumentParser(description='Phase 4: Variance reduction')
    parser.add_argument('--nx', type=int, default=32, help='Grid resolution')
    parser.add_argument('--ny', type=int, default=32)
    parser.add_argument('--nz', type=int, default=32)
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--nmc', type=int, default=64, help='Velocity MC samples')
    parser.add_argument('--nu', type=float, default=0.0, help='Viscosity')
    parser.add_argument('--nd', type=int, default=4, help='Diffusion samples')
    parser.add_argument('--total_time', type=float, default=1.0, help='Total simulation time')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--fps', type=int, default=5, help='Animation FPS')
    parser.add_argument('--no_control_variate', action='store_true', help='Disable control variate')
    parser.add_argument('--no_importance', action='store_true', help='Disable importance sampling')
    args = parser.parse_args()

    # 创建 base 版本（不启用控制变量，也可通过参数切换均匀/重要性）
    sim_base = Simulator4(args.nx, args.ny, args.nz, args.dt, nu=args.nu)
    sim_base.use_control_variate = not args.no_control_variate
    # 若需要切换 importance，可通过修改 BiotSavart3D.importance_sample 逻辑，此处略
    sim_base.nmc = args.nmc
    sim_base.nd = args.nd

    animate_and_report(sim_base, args.total_time, args.output_dir,
                       filename='anim_base.gif', fps=args.fps, label='_base')

    # 如果同时运行 VR 版本（可选），可单独设置参数
    sim_vr = Simulator4(args.nx, args.ny, args.nz, args.dt, nu=args.nu)
    sim_vr.use_control_variate = True   # VR 强制开启
    sim_vr.nmc = args.nmc
    sim_vr.nd = args.nd
    animate_and_report(sim_vr, args.total_time, args.output_dir,
                       filename='anim_vr.gif', fps=args.fps, label='_vr')

if __name__ == '__main__':
    main()