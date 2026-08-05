# Phase 2 README：带自由滑移边界的蒙特卡洛流体模拟

## 算法解决的问题

Phase 2 在 Phase 1 的基础上引入了**固体障碍物**，实现了 **自由滑移边界条件（法向速度为零）**。采用的方法是：

- **流函数法**：引入流函数 $\Psi$，满足 Poisson 方程 $\nabla^2 \Psi = \omega$，边界条件为 $\Psi=0$（单连通边界）。
- **Walk‑on‑Spheres (WoS)**：使用 WoS 算法求解 Poisson 方程，并通过**梯度 WoS** 估计 $\nabla \Psi$，从而恢复速度 $ \mathbf{v} = -\nabla \times \Psi$。
- **对偶采样**：在梯度 WoS 的第一步使用对侧采样降低方差。

该阶段成功模拟了涡旋绕过圆形障碍物的流动，验证了蒙特卡洛方法处理复杂边界的能力。

## 运行代码[main.py](/2022_Simulation/mc_fluid_step2_py/main.py)

1. 进入工程目录：

```bash
cd ./mc_fluid_step2_py
```

2. 运行参数：

```text
--nx                 网格横向分辨率                  默认值: 64
--ny                 网格纵向分辨率                  默认值: 64
--dt                 时间步长                        默认值: 0.1
--nmc                Monte Carlo 采样点数            默认值: 128
--total_time         总模拟总时长                    默认值: 5.0
--output_dir         输出目录                        默认值: ./graph
--fps                输出帧率                        默认值: 10
--obstacle_radius    障碍物半径                      默认值: 0.3
--obstacle_center_x  障碍物中心 x 坐标               默认值: 0.0
--obstacle_center_y  障碍物中心 y 坐标               默认值: 0.0
```

3.运行示例：

```bash
python main.py --nx 64 --ny 64 --nmc 128 --obstacle_radius 0.2 --total_time 5
```

4.运行结果：

[误差分析](/2022_Simulation/mc_fluid_step2_py/output/error_report.txt)

[模拟动画](/2022_Simulation/mc_fluid_step2_py/output/vorticity_with_obstacle.gif)

## 运行代码[compare_mc_vs_grid.py](/2022_Simulation/mc_fluid_step2_py/compare_mc_vs_grid.py)

MC（WoS）与网格流函数法（Grid-based）对比

对比方法选择：采用传统网格流函数法作为“可处理复杂边界”的参考方法。它在规则网格上通过有限差分求解 Poisson 方程（$\nabla^2 \Psi = - \omega$），利用掩码（mask）处理固体边界（$\Psi = 0$），从而支持复杂边界形状。

1. 运行参数：

```text
--nx                 网格横向分辨率                  默认值: 64
--ny                 网格纵向分辨率                  默认值: 64
--dt                 时间步长                        默认值: 0.05
--total_time         总模拟时长（秒）                默认值: 1.0
--nmc                MC WoS路径数                    默认值: 64
--obstacle_radius    障碍物半径                      默认值: 0.3
--obstacle_center_x  障碍物中心 x 坐标               默认值: 0.0
--obstacle_center_y  障碍物中心 y 坐标               默认值: 0.0
--output_dir         输出目录                        默认值: output
```

2. 运行示例

```bash
python compare_mc_vs_grid.py --nx 64 --ny 64 --dt 0.02 --nmc 128 --obstacle_radius 0.2 --total_time 2.0
```

3. 运行结果

[运行结果](/2022_Simulation/mc_fluid_step2_py/output/step2_mc_vs_grid_report.txt)

[可视化](/2022_Simulation/mc_fluid_step2_py/output/step2_mc_vs_grid.png)

4. 结果分析

- Grid Stream Function 方法综合更优：在精度损失可忽略（<4%）的情况下，计算速度提升近两个数量级，内存略增但可接受。对于需要重复模拟或实时性要求的场景，应优先选择 Grid 方法。
- MC 仅在内存敏感或网格生成困难时值得考虑，但需注意其高计算成本。
- MC 与 Grid 在障碍物边界附近的差异较大。MC 在边界附近的误差来自统计噪声与曲率处理，Grid 的误差来自几何离散。改进的核心是分别降低各自的主导误差源，并用收敛性测试确认改进有效。
