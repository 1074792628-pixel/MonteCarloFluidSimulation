import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulator import Simulator
from core.geometry import Geometry
from core.types import Vec2
from render.visualizer import save_animation

def main():
    parser = argparse.ArgumentParser(description='Phase 2: 2D with free-slip obstacles')
    parser.add_argument('--nx', type=int, default=64, help='Grid resolution X')
    parser.add_argument('--ny', type=int, default=64, help='Grid resolution Y')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--nmc', type=int, default=128, help='WoS paths per query')
    parser.add_argument('--total_time', type=float, default=5.0, help='Total simulation time')
    parser.add_argument('--output_dir', type=str, default='graph', help='Output directory')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')
    parser.add_argument('--obstacle_radius', type=float, default=0.3, help='Circle obstacle radius')
    parser.add_argument('--obstacle_center_x', type=float, default=0.0)
    parser.add_argument('--obstacle_center_y', type=float, default=0.0)
    args = parser.parse_args()

    # 创建几何体，可添加更多障碍物（此处仅一个圆形）
    geo = Geometry(outer_bounds=(-1.0, 1.0, -1.0, 1.0))
    geo.add_circle(Vec2(args.obstacle_center_x, args.obstacle_center_y), args.obstacle_radius)

    sim = Simulator(nx=args.nx, ny=args.ny, dt=args.dt, geometry=geo)
    sim.nmc = args.nmc

    save_animation(sim, total_time=args.total_time, output_dir=args.output_dir,
                   filename='vorticity_with_obstacle.gif', fps=args.fps)

if __name__ == '__main__':
    main()