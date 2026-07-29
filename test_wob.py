"""
Test: WoB vs WoS Poisson solver + WoB velocity projection
"""
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba_wos import (
    wos_poisson_grid_numba, wob_poisson_grid_numba,
    wob_project_grid_numba, divergence_2d_numba,
    curl_2d_numba, gradient_2d_numba, bilinear_interp,
    _wob_particular_point, _ray_intersect_rect_numba
)
from scipy.ndimage import gaussian_filter as gs

# ─── Test 1: Poisson solve ∇²p = f ─────────────────────────────────

res = 16
h = 2.0 / (res - 1)
xs = np.linspace(-1, 1, res)
X, Y = np.meshgrid(xs, xs, indexing='ij')

# Source: f(x,y) = -2π² sin(πx) sin(πy)  →  exact p = sin(πx) sin(πy)
f_exact = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
p_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Higher sample counts for WoB
n_part = 512
n_ray = 128

f_grid = f_exact(X, Y)

print("=" * 60)
print("Test 1: WoS vs WoB Poisson Solve")
print(f"Grid: {res}x{res}")
print("=" * 60)

# WoS Poisson
t0 = time.time()
p_wos = wos_poisson_grid_numba(X, Y, f_grid, res,
                                n_walks=512, eps=0.02)
t_wos = time.time() - t0
# Smooth WoS result
p_wos_s = gs(p_wos, sigma=0.4, mode='reflect')
err_wos = np.abs(p_wos_s - p_exact)

# WoB Poisson
t0 = time.time()
p_wob = wob_poisson_grid_numba(X, Y, f_grid, res,
                                n_particular=n_part, n_rays=n_ray)
t_wob = time.time() - t0
# Smooth WoB result
p_wob_s = gs(p_wob, sigma=0.4, mode='reflect')
err_wob = np.abs(p_wob_s - p_exact)

print(f"WoS:  RMSE={np.sqrt(np.mean(err_wos**2)):.6f}, Time={t_wos:.1f}s")
print(f"WoB:  RMSE={np.sqrt(np.mean(err_wob**2)):.6f}, Time={t_wob:.1f}s")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
vmax = max(abs(p_exact).max(), abs(p_wos).max(), abs(p_wob).max())

axes[0,0].imshow(p_exact.T, origin='lower', vmin=-vmax, vmax=vmax,
                  cmap='RdBu_r', extent=[-1,1,-1,1])
axes[0,0].set_title('Exact p')
axes[0,1].imshow(p_wos_s.T, origin='lower', vmin=-vmax, vmax=vmax,
                  cmap='RdBu_r', extent=[-1,1,-1,1])
axes[0,1].set_title(f'WoS p (smoothed)')
axes[0,2].imshow(err_wos.T, origin='lower', cmap='hot',
                  extent=[-1,1,-1,1])
axes[0,2].set_title(f'WoS |error|')

axes[1,0].imshow(p_exact.T, origin='lower', vmin=-vmax, vmax=vmax,
                  cmap='RdBu_r', extent=[-1,1,-1,1])
axes[1,0].set_title('Exact p')
axes[1,1].imshow(p_wob_s.T, origin='lower', vmin=-vmax, vmax=vmax,
                  cmap='RdBu_r', extent=[-1,1,-1,1])
axes[1,1].set_title(f'WoB p (smoothed)')
axes[1,2].imshow(err_wob.T, origin='lower', cmap='hot',
                  extent=[-1,1,-1,1])
axes[1,2].set_title(f'WoB |error|')

plt.tight_layout()
plt.savefig('test_wob_poisson.png', dpi=120)
print("Saved: test_wob_poisson.png\n")

# ─── Test 2: WoB velocity projection ──────────────────────────────

print("=" * 60)
print("Test 2: WoB Velocity Projection")
print("=" * 60)

# Create a divergent velocity field: u = [x, 0], ∇·u = 1
u_div = np.zeros((res, res, 2))
for i in range(res):
    for j in range(res):
        u_div[i, j, 0] = X[i, j]  # u = x
        u_div[i, j, 1] = 0.0

div_before = divergence_2d_numba(u_div[..., 0], u_div[..., 1], h)
mean_div_before = np.mean(np.abs(div_before[1:-1, 1:-1]))

t0 = time.time()
pressure_raw, div_source = wob_project_grid_numba(
    X, Y, u_div[..., 0], u_div[..., 1], res, h,
    n_particular=n_part, n_rays=n_ray)
t_proj = time.time() - t0
# Smooth + gradient in Python
p_smooth = gs(pressure_raw, sigma=0.4, mode='reflect')
px, py = np.gradient(p_smooth, h, h)
u_proj = np.zeros_like(u_div)
u_proj[..., 0] = u_div[..., 0] - px
u_proj[..., 1] = u_div[..., 1] - py

div_after = divergence_2d_numba(u_proj[..., 0], u_proj[..., 1], h)
mean_div_after = np.mean(np.abs(div_after[1:-1, 1:-1]))

print(f"Before projection: mean|div| = {mean_div_before:.6f}")
print(f"After projection:  mean|div| = {mean_div_after:.6f}")
print(f"Reduction factor:  {mean_div_before / max(mean_div_after, 1e-15):.1f}x")
print(f"Time: {t_proj:.1f}s")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].imshow(div_before.T, origin='lower', cmap='RdBu_r',
                  extent=[-1,1,-1,1])
axes[0,0].set_title(f'∇·u BEFORE (mean|div|={mean_div_before:.4f})')
axes[0,1].imshow(div_after.T, origin='lower', cmap='RdBu_r',
                  extent=[-1,1,-1,1])
axes[0,1].set_title(f'∇·u AFTER (mean|div|={mean_div_after:.4f})')
axes[1,0].imshow(p_smooth.T, origin='lower', cmap='viridis',
                  extent=[-1,1,-1,1])
axes[1,0].set_title('Pressure p (WoB smoothed)')
skip = max(1, res//8)
axes[1,1].quiver(X[::skip,::skip], Y[::skip,::skip],
                  u_proj[::skip,::skip,0], u_proj[::skip,::skip,1],
                  scale=1.5, width=0.008)
axes[1,1].set_title('Projected velocity')
axes[1,1].set_aspect('equal')

plt.tight_layout()
plt.savefig('test_wob_projection.png', dpi=120)
print("Saved: test_wob_projection.png\n")

# ─── Test 3: WoB in a full advection-diffusion step ───────────────

print("=" * 60)
print("Test 3: Quick Comparison - WoS vs WoB projection")
print("=" * 60)

from numba_wos import wos_diffuse_grid_numba

# Taylor-Green initial condition
vel0 = np.zeros((res, res, 2))
vel0[..., 0] = -np.cos(np.pi * X) * np.sin(np.pi * Y)
vel0[..., 1] = np.sin(np.pi * X) * np.cos(np.pi * Y)

# Copy for both methods
u_wos = vel0.copy()
u_wob = vel0.copy()

nu = 0.05
dt = 0.03
n_steps = 5

for step in range(n_steps):
    # --- WoS path ---
    du, dv = wos_diffuse_grid_numba(
        X, Y, u_wos[..., 0], u_wos[..., 1], res,
        dt, nu, n_samples=128)
    u_wos[..., 0] = gs(du, sigma=0.3, mode='reflect')
    u_wos[..., 1] = gs(dv, sigma=0.3, mode='reflect')

    # Pressure Poisson projection (WoS)
    div = divergence_2d_numba(u_wos[..., 0], u_wos[..., 1], h)
    div = gs(div, sigma=0.3, mode='reflect')
    p_s = wos_poisson_grid_numba(X, Y, div, res, n_walks=128, eps=0.02)
    p_s = gs(p_s, sigma=0.3, mode='reflect')
    px, py = gradient_2d_numba(p_s, h)
    u_wos[..., 0] -= px
    u_wos[..., 1] -= py
    # No-slip
    for i in range(res):
        for j in range(res):
            if abs(X[i,j]) > 1-1e-6 or abs(Y[i,j]) > 1-1e-6:
                u_wos[i,j] = 0.0

    # --- WoB path ---
    du, dv = wos_diffuse_grid_numba(
        X, Y, u_wob[..., 0], u_wob[..., 1], res,
        dt, nu, n_samples=128)
    u_wob[..., 0] = gs(du, sigma=0.3, mode='reflect')
    u_wob[..., 1] = gs(dv, sigma=0.3, mode='reflect')

    # WoB projection (smooth + gradient in Python)
    p_raw, _ = wob_project_grid_numba(
        X, Y, u_wob[..., 0], u_wob[..., 1], res, h,
        n_particular=n_part, n_rays=n_ray)
    p_s = gs(p_raw, sigma=0.3, mode='reflect')
    px, py = np.gradient(p_s, h, h)
    u_wob[..., 0] -= px
    u_wob[..., 1] -= py
    # No-slip (WoB should handle this, but just in case)
    for i in range(res):
        for j in range(res):
            if abs(X[i,j]) > 1-1e-6 or abs(Y[i,j]) > 1-1e-6:
                u_wob[i,j] = 0.0

    curl_wos = curl_2d_numba(u_wos[..., 0], u_wos[..., 1], h)
    curl_wob = curl_2d_numba(u_wob[..., 0], u_wob[..., 1], h)
    div_wos = np.mean(np.abs(divergence_2d_numba(u_wos[..., 0], u_wos[..., 1], h)))
    div_wob = np.mean(np.abs(divergence_2d_numba(u_wob[..., 0], u_wob[..., 1], h)))

    print(f"Step {step+1}:  WoS|div|={div_wos:.6f}  WoB|div|={div_wob:.6f}")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
curl_wos = curl_2d_numba(u_wos[..., 0], u_wos[..., 1], h)
curl_wob = curl_2d_numba(u_wob[..., 0], u_wob[..., 1], h)
vmax = max(abs(curl_wos).max(), abs(curl_wob).max(), 1e-6)

axes[0,0].imshow(curl_wos.T, origin='lower', cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
axes[0,0].set_title('WoS Projection - Vorticity')
axes[0,1].imshow(curl_wob.T, origin='lower', cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
axes[0,1].set_title('WoB Projection - Vorticity')
axes[0,2].imshow((curl_wos - curl_wob).T, origin='lower', cmap='RdBu_r',
                  extent=[-1,1,-1,1])
axes[0,2].set_title('Difference')

speed_wos = np.sqrt(u_wos[...,0]**2 + u_wos[...,1]**2)
speed_wob = np.sqrt(u_wob[...,0]**2 + u_wob[...,1]**2)
axes[1,0].imshow(speed_wos.T, origin='lower', cmap='viridis',
                  extent=[-1,1,-1,1])
axes[1,0].set_title('WoS |u|')
axes[1,1].imshow(speed_wob.T, origin='lower', cmap='viridis',
                  extent=[-1,1,-1,1])
axes[1,1].set_title('WoB |u|')
axes[1,2].imshow((speed_wos - speed_wob).T, origin='lower', cmap='RdBu_r',
                  extent=[-1,1,-1,1])
axes[1,2].set_title('|u| difference')

plt.tight_layout()
plt.savefig('test_wob_comparison.png', dpi=120)
print("Saved: test_wob_comparison.png")

print("\nDone.")
