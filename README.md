# HighSpeedUUV Kinematic

受速度、加速度、转弯半径和俯仰约束的三维运动学 UUV 规划与跟踪工作空间。

本项目的论文定位是：

> 研究复杂静态障碍环境中的多速度三维路径规划与跟踪，不讨论推进器、空化、
> 螺旋桨效率和高速水动力参数标定。

训练和主评估均使用运动学模型。HoloOcean 仅用于独立场景回放、碰撞几何检查和
可视化，不作为高速动力学验证平台。

## 运行环境

训练和策略评估使用 CUDA GPU。当前锁定环境为 Python 3.10、PyTorch 2.5.1
CUDA 12.4、Stable-Baselines3 2.7.0。首次部署时运行：

```bash
cd /path/to/HighSpeedUUV_Kinematic
./scripts/setup_env.sh
```

脚本会在项目目录创建 `.venv`，安装依赖，并验证 CUDA 与 GPU。若系统缺少
`python3-venv`，请先安装 `python3.10-venv` 或提供可用的 `virtualenv`。

## 运动学模型

状态：

```text
(x, y, z, yaw, pitch, speed, yaw_rate, pitch_rate)
```

策略动作：

```text
[speed_command, yaw_rate_command, pitch_rate_command]
```

约束：

- 速度范围：默认 5–50 节
- 最大加速度和减速度
- 最大偏航角速度
- 最大横向加速度 `|u r|`
- 最小转弯半径
- 最大俯仰角和俯仰角速度
- 偏航/俯仰角加速度限制

这样高速时自动限制转弯角速度，避免“高速大角度急转弯”。

## 观测

```text
sonar  : (6, 12, 48) 三路几何声呐
target : 目标俯仰、偏航、速度误差、归一化距离
path   : Dijkstra 路径俯仰、偏航、前视距离、剩余代价
motion : 当前速度、巡航速度、速度上下限、偏航率、俯仰、俯仰率
```

## 训练

默认流程：

1. 路径跟随教师采集 5,000 步动作；
2. 行为克隆 100 个 epoch，初始化神经网络；
3. BC 模型先做 10 回合多速度评估；
4. 低学习率 PPO 微调；
5. 每 25k 步在 10/20/30/40/50 节上评估；
6. 保留成功率最高的 checkpoint。

```bash
cd /path/to/HighSpeedUUV_Kinematic
./scripts/train.sh --timesteps 500000 --output runs/kinematic_ppo --overwrite
```

训练默认使用 `cuda`、64 路并行采样和 A100 的 TF32 加速。可通过
`--device cuda:0`、`--num-envs 8` 和 `--torch-threads 1` 调整。模型、行为克隆、
策略评估和 PPO 更新均在 GPU 上执行；运动学仿真环境仍由 CPU 并行运行。

训练输出：

- `ppo_kinematic.zip`：周期评估选出的最佳模型
- `ppo_kinematic_final.zip`：PPO 最终模型
- `bc_model.zip`：BC 初始化模型
- `best_eval.json`：最佳模型评估
- `run.json`：完整运动学和训练配置

## 评估

默认分别在 10、20、30、40、50 节各运行 10 回合：

```bash
./scripts/evaluate.sh runs/kinematic_ppo/ppo_kinematic.zip \
  --episodes-per-speed 10 --seed 0 \
  --output runs/kinematic_eval
```

策略评估默认使用 `--device cuda`；如需显式指定 GPU，可传入
`--device cuda:0`。

输出：

- `evaluation_metrics.csv`：逐回合结果
- `evaluation_summary.csv`：按速度分组汇总
- `evaluation_summary.json`
- `evaluation.png`
- `trajectories/*.npz`：可用于 HoloOcean 回放的轨迹

评估同时记录：

- 成功率、碰撞率、越界率、超时率
- 完成时间、路径长度
- 速度跟踪
- 最大横向加速度
- 最大偏航角速度
- 最小障碍物距离
- 最大俯仰角

默认任务成功球半径为 5 m。Dijkstra 规划路径始终以目标球心为终点，
不会再把整个成功球内部都当成零代价区域。

## HoloOcean 几何验证

选择一条导出的轨迹：

```bash
./scripts/replay_holoocean.sh \
  runs/kinematic_eval/trajectories/speed50_seed0.npz
```

回放过程：

- 按相同 NED 场景生成金色障碍物和白色目标球；
- 将运动学轨迹转换为 HoloOcean NWU 坐标；
- 通过智能体位姿接口逐帧回放；
- 检查几何碰撞、最小净距和可视化一致性。

这不验证高速 Torpedo 动力学，只验证场景几何、坐标转换和轨迹结果。

## Sim-to-sim 验证建议

1. 训练环境：本项目的运动学 Gym 环境；
2. 独立验证：不同时间步长、控制延迟和约束扰动的第二套运动学实现；
3. HoloOcean：相同场景的轨迹回放和几何检查；
4. 若需要低速高保真补充，可在 10 节以内使用 Torpedo Fossen 模型对照。

论文中应明确：

> 高速推力、功率、空化和水动力可行性不在本文研究范围内。

## 运动学约束基准

运行速度、加速、转弯和制动基准：

```bash
./scripts/benchmark.sh
```

默认参数下（最大横向加速度 6 m/s²、最小转弯半径 20 m）的结果：

| 目标速度 | 0–90% 加速时间 | 满舵转弯半径 | 制动时间 | 制动距离 |
| -------: | --------------: | -----------: | -------: | -------: |
|    10 节 |          1.50 s |       20.3 m |    1.4 s |    4.1 m |
|    20 节 |          3.40 s |       20.9 m |    3.1 s |   16.9 m |
|    30 节 |          5.20 s |       40.9 m |    4.8 s |   37.9 m |
|    40 节 |          7.10 s |       72.0 m |    6.4 s |   67.2 m |
|    50 节 |          8.90 s |      111.9 m |    8.1 s |  104.9 m |

这些指标直接验证运动学约束，不涉及 RPM、推力、空化或轴功率。

## 测试

```bash
./scripts/test.sh
```
