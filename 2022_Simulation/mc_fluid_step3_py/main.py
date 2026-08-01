import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulator3d import Simulator3D
from render.visualizer3d import save_animation_3d

def main():
    parser = argparse.ArgumentParser(description='Phase 3: 3D Navier-Stokes')
    parser.add_argument('--nx', type=int, default=16, help='Grid resolution X')
    parser.add_argument('--ny', type=int, default=16)
    parser.add_argument('--nz', type=int, default=16)
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--nmc', type=int, default=8, help='Velocity MC samples')
    parser.add_argument('--nu', type=float, default=0.0, help='Viscosity')
    parser.add_argument('--nd', type=int, default=4, help='Diffusion samples')
    parser.add_argument('--total_time', type=float, default=1.0, help='Total simulation time')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--fps', type=int, default=5, help='Animation FPS')
    args = parser.parse_args()

    sim = Simulator3D(nx=args.nx, ny=args.ny, nz=args.nz, dt=args.dt, nu=args.nu)
    sim.nmc = args.nmc
    sim.nd = args.nd

    save_animation_3d(sim, total_time=args.total_time, output_dir=args.output_dir,
                      filename='vorticity_3d_slice.gif', fps=args.fps)

if __name__ == '__main__':
    main()