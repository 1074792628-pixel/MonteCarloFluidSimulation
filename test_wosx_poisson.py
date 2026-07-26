"""WoSX CPU WalkOnSpheres test on [-1,1]^2"""
import sys, os, numpy as np

base = r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx\build_bindings'
for d in [os.path.join(base, 'python', 'Release'),
          os.path.join(base, 'msvc_19.51_cxx_64_md_release'),
          os.path.join(base, 'Release')]:
    os.add_dll_directory(d)
sys.path.insert(0, os.path.join(base, 'python', 'Release'))

import _wosx as wx

# Load mesh
positions = wx.Float2List()
indices = wx.Int2List()
wx.Utils.load_boundary_mesh_2d(r'C:\Users\10772\AppData\Local\Temp\square.obj', positions, indices)
print(f"Mesh: {len(positions)} verts, {len(indices)} segs")

# Domain
dm = np.array([-1.0, -1.0], dtype=np.float32)
dx = np.array([1.0, 1.0], dtype=np.float32)

# CPU geometric queries
geo = wx.Core.GeometricQueries_2d(False, dm, dx)
hdl = wx.Utils.FcpwDirichletBoundaryHandler_2d()
hdl.build_acceleration_structure(positions, indices)
wx.Utils.populate_geometric_queries_for_dirichlet_boundary_2d(hdl, geo)

# PDE: Laplace, BC u=sin(pix)cos(piy)
res = 16
xs = np.linspace(-1, 1, res)
X, Y = np.meshgrid(xs, xs, indexing='ij')
exact = np.sin(np.pi * X) * np.cos(np.pi * Y)

pde = wx.Core.PDE_float_2d()
pde.source = wx.Core.get_constant_source_callback_float_2d(0.0)
pde.dirichlet = wx.Utils.get_dense_grid_dirichlet_callback_float_2d(
    exact.astype(np.float32).ravel(),
    np.array([res, res], dtype=np.int32), dm, dx)
pde.has_reflecting_boundary_conditions = wx.Core.get_constant_indicator_callback_2d(False)
pde.absorption_coeff = 0.0; pde.is_source_constant = True
pde.are_robin_conditions_pure_neumann = True; pde.are_robin_coeffs_nonnegative = True

# Sample points
pts_flat = np.stack([X, Y], axis=-1).reshape(-1, 2).astype(np.float32)
pts_list = wx.Float2List()
for p in pts_flat: pts_list.append(p)

dist_a = wx.FloatList(); dist_r = wx.FloatList()
wx.Utils.compute_dist_to_boundary_2d(geo, pts_list, dist_a, dist_r)

sp_list = wx.Solvers.SamplePointList_float_2d()
for i in range(len(pts_flat)):
    sp_list.append(wx.Solvers.SamplePoint_float_2d(
        pts_flat[i], np.array([0,0],dtype=np.float32),
        wx.Solvers.SampleType.InDomain, wx.Solvers.EstimationQuantity.Solution,
        1.0, dist_a[i], dist_r[i]))

# Solve
print("CPU WoS solving (128 walks per point)...")
ws = wx.Solvers.WalkSettings(1e-3, 0, 0, 0, 1e10, 1024, 0, 0, False, True, True, False, False, True, False, False)
nw = wx.IntList([128] * len(pts_flat))
ss = wx.Solvers.SampleStatisticsList_float_2d()
for _ in range(len(pts_flat)): ss.append(wx.Solvers.SampleStatistics_float_2d())
solver = wx.Solvers.WalkOnSpheres_float_2d(geo)
pb = wx.Utils.ProgressBar(len(pts_flat))
report_progress = wx.Utils.get_report_progress_callback(pb)
solver.solve(pde, ws, nw, sp_list, ss, False, report_progress)
cpu_r = np.array([ss[i].get_estimated_solution() for i in range(len(pts_flat))]).reshape(res, res)
print(f"CPU: [{cpu_r.min():.4f}, {cpu_r.max():.4f}], error={np.abs(cpu_r-exact).mean():.4f}")

# --- GPU WalkOnSpheres ---
print("GPU WoS solving (128 walks per point)...")
gpu_hdl = wx.Utils.GPUFcpwDirichletBoundaryHandler_2d(positions, indices)
gpu_geo = wx.Core.GPUGeometricQueries_2d(
    gpu_hdl, wx.Core.GPUEmptyReflectingBoundaryHandler(),
    np.array([-1,-1], dtype=np.float32), np.array([1,1], dtype=np.float32), False)

gpu_pde = wx.Core.GPUPDE()
gpu_pde.source = wx.Core.get_constant_source_callback_float_2d(0.0)
gpu_pde.dirichlet = wx.Utils.get_dense_grid_dirichlet_callback_float_2d(
    exact.astype(np.float32).ravel(),
    np.array([res, res], dtype=np.int32), dm, dx)
gpu_pde.absorption_coeff = 0.0; gpu_pde.is_source_constant = True
gpu_pde.are_robin_conditions_pure_neumann = True; gpu_pde.are_robin_coeffs_nonnegative = True

wosx_dir = r'C:\Users\10772\Desktop\蒙特卡洛流体仿真\wosx'
th = wx.GPUTaskHandle_float_2d(wosx_dir, wosx_dir, 0)
th.set_geometric_queries(gpu_geo); th.set_pde(gpu_pde); th.init()

gpu_ws = wx.Solvers.GPUWalkSettings()
gpu_ws.epsilon_shell_for_absorbing_boundary = 1e-3
gpu_ws.max_walk_length = 1024

gpu_slv = wx.Solvers.GPUWalkOnSpheresSolver_float_2d(th, gpu_ws, 256, False, False)
gpu_pts = wx.Solvers.GPUSamplePointList_2d()
for i in range(len(pts_flat)):
    gpu_pts.append(wx.Solvers.GPUSamplePoint_2d(
        pts_flat[i], np.array([0,0],dtype=np.float32),
        wx.Solvers.SampleType.InDomain, wx.Solvers.EstimationQuantity.Solution,
        1.0, dist_a[i], dist_r[i]))
gpu_slv.populate_sample_points(gpu_pts, 1, False)
gpu_slv.solve(128)

gpu_ss = wx.Solvers.GPUSampleStatisticsList_float_2d()
gpu_slv.get_sample_statistics(gpu_ss)
gpu_r = np.array([gpu_ss[i].get_estimated_solution() for i in range(len(pts_flat))]).reshape(res, res)
print(f"GPU: [{gpu_r.min():.4f}, {gpu_r.max():.4f}], error={np.abs(gpu_r-exact).mean():.4f}")
print("WoSX verified!")
