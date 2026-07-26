# Monte Carlo 流体仿真方法对比: 2022 vs 2024

## 核心算法差异

| 步骤 | 2022 (Rioux-Lavoie et al.) | 2024 (Sugimoto et al.) |
|------|---------------------------|------------------------|
| **平流** | 半拉格朗日法 (相同) | 半拉格朗日法 (相同) |
| **扩散** | WoS 蒙特卡洛 ∂u/∂t = ν∇²u | WoS 蒙特卡洛 ∂u/∂t = ν∇²u |
| **投影** | 压力泊松方程: ∇²p = ∇·u → u = u* - ∇p | 流函数泊松方程: ∇²ψ = -ω → u = ∇×ψ |
| **状态** | 中间速度 u* → 修正 ∇p | 直接求解流函数 ψ → 重建速度 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `wos_core.py` | 纯 Python WoS 求解器 (慢, 用于教学) |
| `numba_wos.py` | Numba JIT 加速 WoS 求解器 (快 50-100x) |
| `fluid_sim_2022.py` | 2022 方法 (纯 Python) |
| `fluid_sim_2024.py` | 2024 方法 (纯 Python) |
| `fluid_sim_2022_numba.py` | 2022 方法 (Numba 加速) |
| `fluid_sim_2024_numba.py` | 2024 方法 (Numba 加速) |
| `comparison_final.py` | **主对比程序** (推荐使用) |
| `run_comparison.py` | 纯 Python 对比 (旧) |

## 快速运行

```bash
# 最小测试 (10x10 网格, 256 walks, 10 步)
python comparison_final.py --grid 10 --walks 256 --steps 10 --dt 0.02 --nu 0.1

# 高质量测试 (需更长时间)
python comparison_final.py --grid 16 --walks 1024 --steps 20 --dt 0.02 --nu 0.08

# 带圆柱障碍物
python comparison_final.py --grid 12 --walks 512 --steps 15 --obs
```

## 关键参数调优建议

MC walks 数对稳定性和质量影响最大:

| Walks | 速度 | 质量 | 适用场景 |
|-------|------|------|----------|
| 64 | 极快 | 噪声大 | 快速验证 |
| 256 | 快 | 中等 | 初步对比 |
| 1024 | 中等 | 较好 | 正式对比 |
| 4096 | 慢 | 论文级 | 最终结果 |

## GitHub 仓库 GPU 编译指南

### WoSX (nv-tlabs/wosx)
```bash
git clone --recursive https://github.com/nv-tlabs/wosx.git
cd wosx
cmake -S . -B build -DWOSX_ENABLE_GPU_SUPPORT=ON -DWOSX_BUILD_DEMO_APPS=ON
cmake --build build -j4
# Python 绑定
pip install . --force-reinstall
```

### VelMCFluids (rsugimoto/VelMCFluids)
```bash
git clone --recursive https://github.com/rsugimoto/VelMCFluids.git
cd VelMCFluids
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
# 运行: ./velocity_fluids_divfree_advection ../configs/config_cohomology.json
# 可视化: python ../visualize.py ../results/results_cohomology
```

## 方法论说明

**2022 方法** 的核心是算子分裂:
1. 平流: 用半拉格朗日法更新速度场
2. 扩散: 用 Walk-on-Spheres 求解热方程 ∂u/∂t = ν∇²u
3. 投影: 求解压力泊松方程 ∇²p = ∇·u*, 然后修正 u = u* - ∇p

**2024 方法** 的核心创新:
1. 平流: 同上
2. 扩散: 同上 (用 Walk-on-Boundary)
3. 投影: 用流函数 ∇²ψ = -ω, 然后 u = ∇×ψ
   - 自动满足 ∇·u = 0
   - 避免了压力求解中的 Neumann 边界条件处理
   - 在 GPU 实现中更加高效 (可直接在速度场上操作)

## 局限性 (Python 实现)

1. **MC 噪声**: 有限 walk 数导致 PDE 求解有噪声, 梯度算子放大噪声
2. **速度**: Python 循环比 GPU/C++ 慢 10⁴-10⁶ 倍
3. **稳定性**: 噪声累积可能导致能量增长 (非物理)
4. **边界条件**: 压力泊松用 Dirichlet BC 近似 (正确应为 Neumann)
