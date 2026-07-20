# 正式实验结果快照（2026-07-20）

本文件只固化由 `experiments/run_paper_experiments.ps1` 自动生成并经脚本
复核的聚合结果。原始数据、逐样本预测、模型权重和拟合后的校准器均不
进入 Git。完整本地产物位于 `artifacts/runs/`。

## 1. 冻结协议

- 正式数据集只有两个：SyncG 官方 train/test 与 RPM-10K
  single-pointer subset。
- SyncG 使用固定 Hugging Face 提交
  `14204c3f5b35d160fafa39ad195cd5a63e6e9c12`。
- SyncG train/test 分别为 16,000/4,000 张；train 样本 ID 哈希为
  `6c1bcfd7a6a83c07e6a8d6c133f0fc48abed543ca02d02c6ee46804240216e75`。
- RPM-10K 从 2,000 张测试图中按预声明规则固定保留 1,797 张单指针、
  标量且量程内样本。RPM 标签不用于训练、阈值选择或模型选择。
- 所有读数方法共享同一份冻结前端缓存。失败样本在 NMAE 中按 1.0
  惩罚，在准确率中计错，同时单独报告 coverage。
- 残差与门控只在 SyncG train 上进行 5 折嵌套分组交叉拟合；
  `group_id` 为表型与场景组合，测试组不参与残差裁剪、门控标签或阈值。
- 正式随机种子为 `20260720`，图像校正固定为 `off`。

## 2. 训练环境与检查点

| 项目 | 值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 4060 |
| U2NetP 训练 | 15 epochs，batch 8，AMP，14,375 train / 1,625 validation |
| 最佳 epoch | 15 |
| 原始概率验证 Dice | 0.915483 |
| 部署后处理 | 去 Letterbox 后保留最大连通域 |
| 部署阈值 / 验证 Dice | 0.7 / 0.914163 |
| 微调检查点 SHA-256 | `3d7523933c54666d2e594eb6d0cb48781974b067992a16d889726ae4524700e1` |
| 正式运行签名 | `c2237a68fd801055ef097a39c3b7d9a7816f51e6199925e025a97fc75a4ed81c` |

15 轮共 26,955 个批次，其中 AMP 动态缩放保护性跳过 4 次
（0.0148%）；没有 NaN、OOM 或训练中断。

## 3. 分割组件结果

该表使用 SyncG test 的真值表盘框，只评估分割组件，不属于端到端主表。
两套权重的阈值均预先在相同的 SyncG train-validation 上选择，test
不重新搜索阈值。

| 分割前端 | 阈值 | Micro Dice | Micro IoU | Macro Dice | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Released calibrated | 0.004 | 0.248642 | 0.141971 | 0.197238 | 0.258522 | 0.239490 |
| SyncG fine-tuned | 0.7 | 0.912560 | 0.839183 | 0.898478 | 0.916592 | 0.908564 |

微调带来 0.663918 的绝对 micro-Dice 增益。该结果支持“分割域适配有效”，
但不能替代端到端读数实验。

## 4. 端到端主表

| Method | SyncG NMAE ↓ | SyncG Acc@2% ↑ | SyncG coverage ↑ | RPM NMAE ↓ | RPM Accε@1% ↑ | RPM Accθ@5% ↑ | RPM coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Transformer | 0.2062 | 0.1792 | 0.9925 | 0.5260 | 0.0306 | 0.0249 | 0.7273 |
| Geometry-v1 | 0.1570 | 0.3033 | 0.9925 | 0.5302 | 0.0428 | 0.0307 | 0.7273 |
| Geometry-v2 | 0.1568 | 0.3003 | 0.9925 | 0.5302 | 0.0428 | 0.0325 | 0.7245 |
| Mean Fusion | 0.1565 | 0.3038 | 0.9925 | 0.5255 | 0.0456 | 0.0360 | 0.7273 |
| Quality-weighted Fusion | 0.1565 | 0.3040 | 0.9925 | 0.5255 | 0.0462 | 0.0371 | 0.7273 |
| Residual without Gate | **0.1149** | 0.4195 | 0.9925 | **0.4966** | 0.0150 | 0.0296 | 0.7273 |
| Ours (selective gate) | 0.1152 | **0.4315** | 0.9925 | 0.4973 | 0.0200 | 0.0342 | 0.7273 |

RPM 指标只对应已知量程的 1,797 张 single-pointer subset，不是完整
DialBench 2,000 张官方榜单结果。

## 5. 配对统计结论

所有区间均按 `group_id` 进行 bootstrap：

| 比较 | 数据集 | ΔNMAE（candidate - baseline） | 95% CI | 结论 |
|---|---|---:|---:|---|
| Ours vs Transformer | SyncG | -0.091053 | [-0.097089, -0.084580] | 显著改善 |
| Ours vs Transformer | RPM | -0.028671 | [-0.043307, -0.010092] | 冻结外测仍改善 |
| Ours vs Quality-weighted | SyncG | -0.041315 | [-0.043288, -0.039443] | 显著改善 |
| Ours vs Quality-weighted | RPM | -0.028109 | [-0.036935, -0.016413] | 冻结外测仍改善 |
| Quality-weighted vs Mean | SyncG | -0.000020 | [-0.000049, -0.000001] | 极小但区间不跨 0 |
| Quality-weighted vs Mean | RPM | -0.000056 | [-0.000153, 0.000038] | 不显著 |
| Ours vs Residual no gate | SyncG | +0.000305 | [0.000010, 0.000657] | 门控 NMAE 略差 |
| Ours vs Residual no gate | RPM | +0.000707 | [-0.000994, 0.002342] | 不显著 |

因此不能声称质量加权或门控在所有数据集上普遍降低平均误差。主要精度
贡献来自几何/Transformer 互补与跨量程归一化残差。

## 6. 门控与特征消融

| Variant | NMAE ↓ | Acc@2% ↑ | Correction coverage ↑ | Negative transfer ↓ |
|---|---:|---:|---:|---:|
| Quality-weighted（无残差） | 0.1565 | 0.3040 | — | — |
| Residual without Gate | **0.1149** | 0.4195 | 0.9925 | 0.2872 |
| Full Ours | 0.1152 | 0.4315 | 0.7330 | 0.1944 |
| Geometry features only | 0.1162 | 0.4285 | 0.7005 | 0.1888 |
| No mask-quality features | 0.1150 | **0.4348** | 0.7235 | **0.1862** |
| No ellipse features | 0.1152 | 0.4333 | 0.7360 | 0.1974 |
| Disagreement only | 0.1478 | 0.2820 | 0.2805 | 0.3734 |

完整门控把无条件残差的负迁移率从 28.72% 降到 19.44%，并提高
Acc@2%，但平均 NMAE 略差。去掉 mask-quality 特征的变体更好，表明
这些特征存在冗余或合成域偏差；建议正文报告，后续再做特征选择与校准，
不要从本次 test 结果反向调参。

## 7. 失败归因与合成到真实迁移

| 数据集 | N | Success | Meter not found | Pointer not found |
|---|---:|---:|---:|---:|
| SyncG test | 4,000 | 3,970 (99.25%) | 27 | 3 |
| RPM single-pointer | 1,797 | 1,307 (72.73%) | 28 | 462 |

RPM 的主要失败来源是指针分割/中心线校验，而不是读数后端异常。

只替换分割前端、保持读数头和全部配置不变时：

| 分割前端 | Transformer NMAE | Weighted NMAE | Coverage |
|---|---:|---:|---:|
| Released | 0.7943 | 0.7919 | 0.3428 |
| SyncG fine-tuned | 0.5260 | 0.5255 | 0.7273 |

SyncG 分割微调把 RPM 端到端 coverage 提高 38.45 个百分点，但绝对
NMAE 仍接近 0.5。论文应如实指出真实域上的端点语义、刻度范围和图像
分布仍是主要瓶颈。

## 8. 推荐论文表述

可以主张：

1. 修正 Letterbox 坐标映射并在 SyncG train 上微调轻量分割前端，
   显著提高同域分割及真实域端到端覆盖率。
2. 双几何估计与 Transformer 提供互补读数，跨量程归一化残差在同域和
   冻结外测上均显著降低 NMAE。
3. 选择性门控提供可解释的风险—覆盖率权衡，降低负迁移并提高小误差
   命中率，但不是平均 NMAE 的最优方案。
4. 嵌套分组 OOF、全分母失败惩罚、配对分组 bootstrap 与冻结外测共同
   构成防泄漏的务实评测协议。

不能主张：

- 完整 RPM-10K / DialBench 排行榜成绩；
- 在 RPM 上训练、微调或选择阈值；
- 门控或质量加权在所有指标、所有域上都优于简单方案；
- state of the art；
- 已解决真实域泛化问题。

## 9. 本地正式产物

- `artifacts/runs/main_table.md`
- `artifacts/runs/ablation_table.md`
- `artifacts/runs/failure_table.md`
- `artifacts/runs/frontend_transfer_table.md`
- `artifacts/runs/risk_coverage.png` 与 `.pdf`
- `artifacts/runs/syncg_segmentation/syncg_test_metrics.json`
- `artifacts/runs/syncg_test/metrics.json`
- `artifacts/runs/rpm10k_single_pointer_zero_shot/metrics.json`

最终审计：自动测试 21/21 通过；三份正式 manifest、四份冻结预测缓存与
上述论文产物均通过数量和存在性校验。
