# Phase 1 README：2D 无粘蒙特卡洛流体模拟

## 算法解决的问题

Phase 1 实现了**二维不可压缩无粘欧拉方程的蒙特卡洛求解器**，作为整篇文献复现的基础原型。核心思路是：

- 使用 **涡量-速度（vorticity-velocity）公式**，通过 Biot‑Savart 定律由涡量场重建速度场。
- 采用 **半拉格朗日后向追踪** 平流涡量。
- 利用 **网格缓存** 存储上一时间步的涡量场，避免递归导致的指数复杂度。
- 使用 **均匀蒙特卡洛采样** 估计 Biot‑Savart 积分。

该原型在无边界开放域中模拟了两个高斯涡旋的相互绕转，验证了后向追踪 + MC 估计 + 缓存这一基本流程的可行性。

## 运行[main.py](/2022_Simulation/mc_fluid_step1_py/main.py)代码

1. 进入工程目录：

```bash
cd ./mc_fluid_step1_py
```

2. 运行参数：

```text
--nx                 网格横向分辨率                  默认值: 64
--ny                 网格纵向分辨率                  默认值: 64
--dt                 时间步长                        默认值: 0.1
--nmc                Monte Carlo 采样点数            默认值: 256
--total_time         总模拟时长（秒）                默认值: 10.0
--output_dir         输出目录                        默认值: ./graph
--fps                输出动画帧率                    默认值: 20
```

3.运行示例：

```bash
python main.py --nx 64 --ny 64 --dt 0.1 --nmc 128 --total_time 5
```

4. 输出结果

[输出动画](/2022_Simulation/mc_fluid_step1_py/output/vorticity.gif)

## 运行[compare_tg.py](/2022_Simulation/mc_fluid_step1_py/compare_tg.py)代码

本方案利用 Taylor‑Green 涡旋(解析表达式：$\omega(x,y) = 2 \cos(\pi x) \cos(\pi y)$)
作为标准参考解（解析解），通过标准库 numpy 计算其精确涡量场，并与 Step 1 的蒙特卡洛模拟结果对比，量化 MC 方法的 数值耗散误差 和 统计方差。误差数据与曲线将输出至 mc_fluid_step1_py/output/ 文件夹。

1. 运行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--nx` | int | 64 | 网格横向分辨率 |
| `--ny` | int | 64 | 网格纵向分辨率 |
| `--dt` | float | 0.05 | 时间步长 |
| `--nmc` | int | 256 | Monte Carlo 采样点数 |
| `--total_time` | float | 1.0 | 总模拟时长（秒） |
| `--output_dir` | str | "output" | 输出目录 |

2. 运行示例

```bash
python compare_tg.py --nx 128 --ny 128 --dt 0.02 --nmc 512 --total_time 2 
```

3. 输出结果

[误差曲线](/2022_Simulation/mc_fluid_step1_py/output/error_curve.png)

[误差分析](/2022_Simulation/mc_fluid_step1_py/output/error_report.txt)

4. 结果分析

- $L_2$ 相对误差随时间近似线性增长：从 0 增长到约 0.157（2.0 s），增幅稳定，未出现指数发散，说明数值格式在该网格与时间步长下是稳定的.
- $L_{\infty}$误差呈现显著波动，但整体也呈增大趋势（从约 0.1 到约 0.5，峰值可达 1.5）。其数值远高于 L2 误差，表明误差分布不均匀，存在局部大误差区域.
- 该计算在 2 秒内保持稳定，误差呈可控增长，适合作为涡旋演化的基准测试.
- 当前误差水平较高（2 s 时 L2 达 15.7%），若需更高保真度，应加密网格或采用更低耗散的格式（如高阶 WENO/紧致格式）。


## 运行[study_nmc.py](/2022_Simulation/mc_fluid_step1_py/study_nmc.py)代码

本代码研究误差随样本数 $n$ 的变化

研究步骤

- 固定参数：选择网格 $64 \times 64$，dt=0.02，总时间 t=1.0s。
- 遍历样本数：取 $n = 16, 32, 64, 128, 256, 512$。
- 对每个 $n$ 运行模拟，在 t=1.0s 处记录 L2 误差。
- 多次重复取平均（例如每个 $n$ 运行 5 次取平均），以消除单次随机波动。
- 绘制 log–log 图，拟合斜率。

1. 运行参数

```text
--nx                 网格横向分辨率                  默认值: 64
--ny                 网格纵向分辨率                  默认值: 64
--dt                 时间步长                        默认值: 0.02
--total_time         总模拟时长（秒）                默认值: 1.0
--nmc_list           MC样本数列表（逗号分隔）        默认值: 16,32,64,128,256,512
--repeats            每个样本数重复运行次数          默认值: 5
--output_dir         输出目录                        默认值: ./output
```

2. 运行示例

```bash
python study_nmc.py --nx 64 --dt 0.02 --nmc_list "8,32,64,128,256" --repeats 3
```

3. 运行结果

[误差数据](/2022_Simulation/mc_fluid_step1_py/output/nmc_study.txt)

[误差图表](/2022_Simulation/mc_fluid_step1_py/output/nmc_error_convergence.png)

4. 结果分析

- 增加样本数只能有限地降低误差。当 n 从 128 增至 512 时，平均 L2 误差仅从 0.0899 降至 0.0846，改善不足 6%；而计算量增加了 4 倍。
- 误差中系统误差（离散误差/格式耗散）占主导，随机采样误差已经很小。log-log 斜率偏离理论值 -0.5，进一步说明蒙特卡洛样本数并非当前精度的瓶颈。
- 若计算资源有限，n = 128 已经能提供较低且稳定的误差，继续增加 n 的性价比很低。

## 运行[compare_with_particles.py](/2022_Simulation/mc_fluid_step1_py/compare_with_particles.py)

对比实验实现：MC 方法 vs 涡量粒子法（Taylor-Green 基准）

- 使用 NumPy/SciPy（现有库）实现涡量粒子法；
- 以 Taylor-Green 涡旋 作为基准（无粘稳态解，参考场 = 初始场）；
- 将两种方法输出到统一网格，计算 L2 相对误差；
- 记录 L2 误差、计算时间、内存；
- 绘制 误差-时间曲线 和 误差-计算时间 Pareto 曲线。

1. 运行参数

```text
--nx                 网格横向分辨率                  默认值: 32
--ny                 网格纵向分辨率                  默认值: 32
--dt                 时间步长                        默认值: 0.02
--total_time         总模拟时长（秒）                默认值: 0.5
--nmc_list           MC样本数列表（逗号分隔）        默认值: 16,64,256,1024
--output_dir         输出目录                        默认值: output
```

2. 运行示例

```bash
python compare_with_particles.py --nx 64 --ny 64 --dt 0.01 --total_time 1.0
```

3. 运行结果

[输出结果](/2022_Simulation/mc_fluid_step1_py/output/comparison_report.txt)

[误差-时间曲线](/2022_Simulation/mc_fluid_step1_py/output/error_vs_time.png)

[Pareto 曲线](/2022_Simulation/mc_fluid_step1_py/output/pareto_curve.png)

4. 结果分析

- Vortex Particle 显著优于 MC：在几乎相同的误差水平下，计算时间比 MC(nmc=1024) 快两个数量级，比 MC(nmc=256) 快约 8 倍，同时精度更高。因此对于此基准问题，应优先采用 Vortex Particle 方法。
- MC 的样本数存在“有效上限”：当 n 超过约 256 后，误差几乎不降，继续增加样本仅浪费计算时间。若必须使用 MC，n≈128–256 即可达到性价比平衡点。
