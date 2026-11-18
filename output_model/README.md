# Output Model

`output_model` 只保留当前正式训练、评估和可视化结果。

```text
output_model/
├── kinematic_ppo_v2/         # 50 万步训练结果、模型和训练配置
│   ├── csv/
│   │   ├── progress.csv      # PPO 训练曲线
│   │   └── monitors/         # 各并行环境回合记录
│   ├── bc_model.zip
│   ├── best_model.zip
│   ├── ppo_kinematic.zip
│   ├── ppo_kinematic_final.zip
│   └── run.json
├── kinematic_eval_v2/        # 当前模型的分速度评估
│   ├── csv/
│   │   ├── evaluation_metrics.csv
│   │   └── evaluation_summary.csv
│   ├── evaluation.png
│   ├── evaluation_summary.json
│   └── trajectories/
└── visualization/            # 五档速度诊断图和 50 节实时回放
    ├── speed10_seed0_diagnostics.png
    ├── speed20_seed0_diagnostics.png
    ├── speed30_seed0_diagnostics.png
    ├── speed40_seed0_diagnostics.png
    ├── speed50_seed0_diagnostics.png
    ├── speed50_seed0_playback.mp4
    └── visualization_summary.json
```

默认情况下，训练、评估和基准程序的 CSV 都写入其 `--output` 目录下的
`csv/`；并行环境记录额外放在 `csv/monitors/`。
