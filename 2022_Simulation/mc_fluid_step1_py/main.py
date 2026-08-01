import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulator import Simulator
from render.visualizer import save_animation

def main():
    parser = argparse.ArgumentParser(description='Phase 1: 2D inviscid Monte Carlo fluid')
    parser.add_argument('--nx', type=int, default=64, help='Grid resolution X')
    parser.add_argument('--ny', type=int, default=64, help='Grid resolution Y')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--nmc', type=int, default=256, help='MC samples per velocity query')
    parser.add_argument('--total_time', type=float, default=10.0, help='Total simulation time')
    parser.add_argument('--output_dir', type=str, default='graph', help='Output directory')
    parser.add_argument('--fps', type=int, default=20, help='Animation FPS')
    args = parser.parse_args()

    sim = Simulator(nx=args.nx, ny=args.ny, dt=args.dt)
    sim.nmc = args.nmc

    save_animation(sim, total_time=args.total_time, output_dir=args.output_dir,
                   filename='vorticity_evolution.gif', fps=args.fps)

if __name__ == '__main__':
    main()