# Phase 3 README：3D Navier‑Stokes 蒙特卡洛流体模拟

## 算法解决的问题

Phase 3 将模拟从 2D 扩展到 **3D 不可压缩 Navier‑Stokes 方程**，核心贡献是：

- 使用 **Feynman‑Kac 公式** 将粘性和涡拉伸表示为随机过程的期望。
- 算法分为三个子步骤：
  1. **平流**：半拉格朗日后向追踪（同 Phase 1）。
  2. **扩散**：在平流位置上叠加高斯随机扰动，模拟 Wiener 过程。
  3. **涡拉伸**：使用涡量分段法估计应变率张量作用。
- 采用均匀 3D 网格缓存存储涡量（向量场）和速度场，避免递归。
- 输出 z=0 切面的涡量幅值动画。

该阶段首次将 3D NS 方程的蒙特卡洛求解在图形学领域实现，验证了 Feynman‑Kac 公式的实际可行性。

## 运行代码[main.py](/2022_Simulation/mc_fluid_step3_py/main.py)

1. 进入工程目录：

```bash
cd ./mc_fluid_step3_py
```

2. 运行参数：

```text
--nx                 网格 x 方向分辨率               默认值: 16
--ny                 网格 y 方向分辨率               默认值: 16
--nz                 网格 z 方向分辨率               默认值: 16
--dt                 时间步长                        默认值: 0.1
--nmc                Monte Carlo 采样点数            默认值: 8
--nu                 运动粘性系数                    默认值: 0.0
--nd                 扩散子步采样数                  默认值: 4
--total_time         总模拟时长（秒）                默认值: 1.0
--output_dir         输出目录                        默认值: ./output
--fps                输出动画帧率                    默认值: 5
```

3.运行示例：

```bash
python main.py --nx 32 --ny 32 --nz 32 --dt 0.1 --nu 0.0 --nmc 32 --nd 4
```

## 运行[study_nmc.py](/2022_Simulation/mc_fluid_step3_py/study_nmc3d.py)

对 3D Taylor-Green 涡旋测试不同 nmc 下的 L2 相对误差、标准差，并绘制 log-log 收敛曲线。

1. 运行参数

```text
--nx                 网格 x/y/z 分辨率              默认值: 16
--dt                 时间步长                        默认值: 0.02
--total_time         总模拟时长（秒）                默认值: 0.5
--nu                 运动粘性系数                    默认值: 0.01
--nmc_list           MC 样本数列表（逗号分隔）       默认值: 16,32,64,128,256
--repeats            每个样本数重复次数              默认值: 3
--output_dir         输出目录                        默认值: output
```

2. 运行示例

```bash
python study_nmc3d.py --nx 32 --nmc_list "32,64,128,256,512" --repeats 3 --total_time 1.0
```

3. 运行结果

[运行图表](/2022_Simulation/mc_fluid_step3_py/output/step3_nmc_error_convergence.png)

[运行数据](/2022_Simulation/mc_fluid_step3_py/output/step3_nmc_study.txt)

4. 结果分析

- 统计误差已被有效控制，误差以系统误差为主；
- 系统误差可能来源于时间/空间离散、边界处理或参考解匹配问题；

## 运行[compare_burgers3d.py](/2022_Simulation/mc_fluid_step3_py/compare_burgers3d.py)

1. 运行参数

```text
--nx, --ny, --nz      网格分辨率  32 
--dt                  时间步长  0.02 
--total_time          总模拟时长  1.0 
--nmc                 MC 速度采样数  64 
--nu                  运动粘性系数  0.05 
--a                   Burgers 涡拉伸参数  1.0 
--Gamma               环量强度  1.0 
--L                   域半边长（[-L,L]³）  1.0 
--output_dir          输出目录  output 
--fps                 动画帧率  10 
```

2. 运行示例

```bash
python compare_burgers3d.py --nx 32 --ny 32 --nz 32 --nmc 128 --a 1.0 --Gamma 1.0 --nu
 0.01 --total_time 2.0
```

3. 运行结果

[模拟动画](/2022_Simulation/mc_fluid_step3_py/output/step3_burgers_evolution.gif)

[模拟数据](/2022_Simulation/mc_fluid_step3_py/output/step3_burgers_error_curve.png)

4. 结果分析

- 当前 MC 模拟器在 Burgers 涡上的误差主要源于数值耗散（网格插值）和速度方差，而非物理粘性破坏稳态。即使初始场与解析解完全一致，粗网格和有限采样仍无法维系拉伸-扩散平衡，导致涡量快速衰减。

- 这意味着单纯提高 nmc 的收益有限，核心需要提高网格分辨率并配合方差缩减技术，才能使误差降至可接受范围（如 <5%）
