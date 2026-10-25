# Image structure source control

**已完成；科学结论为 `NOT_RETAINED`，`claim_supported=false`。** 固定 30E、最终 `last30` 的 image-structure 表示未通过两项 native learning 门槛和五项 native 联合门槛。候选 native 的 mean / worst-30 mean / maximum 为 **23.1894797459 / 55.7430449128 / 68.6384409666%FS**，`>5%FS` 为 **239/292**；identity15 参照为 **2.54312182437 / 11.8009553314 / 53.7970364094%FS**，`>5%FS` 为 **16/292**。

本轮关闭这一固定表示与固定 30E source control。整体研究状态仍为 **ACTIVE**。本结果不触发 promotion、长训、额外真实域评估，或为通过门槛而调整 kernel、scale、padding、mixture、loss、epochs、seed、输入、门槛或 cohort。

科学修订：`d6b2db442a1d89799dff4024cd6baa26c0f2d28c`；执行修订：`3263eaf90c4b985f78c72ab7484e1dcac6e3c158`。两阶段串行任务均以 exit 0 完成；source 阶段结束于 `2026-10-25T21:51:07.564473+00:00`。保存的 `2026-10-25T22:05:46.807129+00:00` completion/idle 记录显示三个实际父子进程均已退出，GPU compute rows 为空。

## Native 结果与冻结门槛

所有标量结果沿用提供的标签，保留同一组 292 个 sample ID、73 个每组四图的 group ID。四份保存的标量状态均为 coverage 1、输入/输出失败 0。预测范围包含端点 0 和 1；失败样本的 error=1 保留在完整 292 分母中。尾部定义为最差 `ceil(292*0.1)=30` 图的平均误差。

| 指标 | identity15 native | image-structure30 native | constant 0.5 |
|---|---:|---:|---:|
| Mean %FS | 2.54312182437 | 23.1894797459 | 25.0089274061 |
| Median %FS | 1.54945105314 | 19.3514585495 | 24.896581471 |
| P90 %FS | 4.09151151776 | 48.632774353 | 44.1684859991 |
| P95 %FS | 5.40433466434 | 54.5679415762 | 47.6235611923 |
| Worst-30 mean %FS | 11.8009553314 | 55.7430449128 | 47.3844161057 |
| Maximum %FS | 53.7970364094 | 68.6384409666 | 49.9422221968 |
| Acc@1 count /292 | 102 | 13 | 7 |
| Acc@2 count /292 | 182 | 28 | 14 |
| Acc@5 count /292 | 276 | 53 | 24 |
| Errors >1%FS | 190 | 279 | 285 |
| Errors >2%FS | 110 | 264 | 278 |
| Errors >5%FS | 16 | 239 | 268 |

Learning gate 要求 mean 不大于 constant 的一半（12.504463703%FS），worst-30 mean 不大于 constant 的一半（23.6922080529%FS）；两项均 FAIL。Native 联合门槛要求同时满足 mean ≤ identity native、tail 严格小于 identity native、maximum ≤ identity native、`>5%FS` ≤16，以及参照 native error≤1%FS 的 102 图中没有新增 `>5%FS`；五项均 FAIL。

逐图比较：26 改善、266 恶化、0 不变；逐组平均误差比较：0 改善、73 恶化、0 不变。参照原本 error≤1%FS 的 102 图中，83 图在候选中成为 `>5%FS`。

| 门槛 %FS | 新增超限 | 救回 | 参照超限 | 候选超限 |
|---|---:|---:|---:|---:|
| 1 | 97 | 8 | 190 | 279 |
| 2 | 165 | 11 | 110 | 264 |
| 5 | 224 | 1 | 16 | 239 |

最终 epoch 30 保存的训练 SmoothL1 为 `0.006354563796670438`，训练 NMAE 为 `1.760571110543552%FS`。Native learning gate 使用上述 held-out native 结果。

## 条件停止与参照 CF

Native 失败后正常条件停止：候选 CF `eligible=false`、`candidate_result=null`、0 images、0 calls，且无候选 CF 结果文件。候选 CF、source03、真实 Industrial/RF 均未执行；source03/真实域 RGB、model forward、backward、update 均为 0。

归档中的 `reference_source_counterfactual_*` 是已有 identity15 的 raw RG-swap `[1,0,2]` 参照，其自身 error≤1%FS 集合为 104 图；它不构成本轮候选 CF 观察。该参照 mean / worst-30 mean / maximum 为 `2.6353762864682597 / 12.945961876345487 / 54.24728989601135%FS`，`>5%FS` 为 17/292。冻结的候选 CF 联合门槛原要求 mean/tail/max 均不劣于该参照、`>5%FS`≤17，且其自身 104 图准确集合没有新增 `>5%FS`；原 color drift 阈值只作诊断。

## 执行与成本

固定 RGB-derived 梯度结构测量位于 reader 内、原 encoder 前；完整 MNV4 与原 batch-mean SmoothL1 beta=.05 训练配方保留。模型有 8,435,793 个 learned parameters、233 个可训练 parameter tensors、465 个完整 states；其中新增一个固定 F32 kernel buffer，没有新增可训练参数。固定滤波器是额外 F32 compute，其 event/check 开销计入相应 phase/full wall。

| 范围 | 图像呈现 | CNN / point / fixed-filter 调用，各自 | backward attempts | optimizer updates | AMP skips |
|---|---:|---:|---:|---:|---:|
| 正式 30E 训练 | 164880 | 2580 | 2580 | 2577 | 3 |
| 正式 native | 292 | 5 | 0 | 0 | 0 |
| 正式合计 | 165172 | 2585 | 2580 | 2577 | 3 |
| 独立 preflight | 8 unique RGB | 3 | 1 combined | 0 | — |

每个 epoch 86 batches、5496 图；固定 30E、无中途验证、scheduler、best 选择或延长。Identity15 参照的训练呈现为 82440 图、1290 backward attempts；本轮为 164880 图、2580 attempts，预先冻结的计划训练预算为 2x。

| 时间/内存范围 | 保存值 |
|---|---:|
| 训练 phase wall | 885.743702851 s |
| Native pipeline / phase wall | 5.0427265726 / 5.04454496503 s |
| Native full-model forward | 3.5219172462821007 s |
| Filter event time：训练 / native / 正式总计 | 16.5573310032 / 0.0288711681366 / 16.5862021713 s |
| 正式 full process wall | 900.88052877 s |
| Supervisor source stage / whole pipeline | 902.043468833 / 923.498824764 s |
| 正式 CUDA peak allocated / reserved | 8398941696 / 11058282496 bytes |
| 独立 point-forward time | null |
| Preflight full process wall / supervisor stage | 18.6658721007 / 19.5334707238 s |
| Preflight filter event time | 0.101651362896 s |
| Preflight CUDA peak allocated / reserved | 631291392 / 734003200 bytes |

Preflight 单列于正式计数之外：8 unique RGB、3 CNN/point/filter calls、1 combined backward、0 optimizer updates。另有 4 张 ID 图为 CPU native-input identity witness 重新解码，其读取/检查包含在 initialization/full wall 中；未保存独立 CPU identity-check 用时。Point 独立用时保持 `null`。

## 已完成审计与归档内容

本轮唯一 fresh、独立非作者的 standard-library saved-number/execution CPU audit 给出 **`PASS_WITH_QUALIFICATIONS`**：81 PASS、7 scientific FAIL、0 integrity FAIL。11698 个数值比较无不一致，最大绝对差 `1.4210854715202004e-14`；浮点绝对容差为 `1e-12`，整数、ID 与 group 为精确比较。科学结论为 `NOT_RETAINED`。

`completed/` 仅保存该审计明确允许公开的 17 个文件投影：14 个 CSV 统一 LF 并以 `.csv.gz` 无损保存，3 个 JSON/Markdown 原文保存。CSV 解压后的文件名对应审计报告中的 `.csv` 名称。所有原始数值字符串、ID/group、负面结果与 null 均保留；本次归档不重新计算指标。实际公开文件逐一列于 [public_allowlist.json](public_allowlist.json)。

| 内容 | 文件 |
|---|---|
| 科学与 integrity 结论、逐项检查 | [audit_result.json](completed/audit_result.json)、[audit_report.md](completed/audit_report.md)、[checks.csv.gz](completed/checks.csv.gz) |
| 保存数值比较 | [numeric_comparisons.csv.gz](completed/numeric_comparisons.csv.gz)、[metric_comparisons.csv.gz](completed/metric_comparisons.csv.gz) |
| Native 逐图、逐组及门槛配对 | [native_paired_rows.csv.gz](completed/native_paired_rows.csv.gz)、[native_paired_groups.csv.gz](completed/native_paired_groups.csv.gz)、[native_threshold_comparisons.csv.gz](completed/native_threshold_comparisons.csv.gz) |
| 允许的元数据投影 | [collected_metadata.json](completed/collected_metadata.json) |
| 本轮 native 与 constant 的逐图/逐组标量 | [current_source_id_predictions.csv.gz](completed/current_source_id_predictions.csv.gz)、[current_source_id_groups.csv.gz](completed/current_source_id_groups.csv.gz)、[current_constant_source_id_predictions.csv.gz](completed/current_constant_source_id_predictions.csv.gz)、[current_constant_source_id_groups.csv.gz](completed/current_constant_source_id_groups.csv.gz) |
| 既有 identity15 native 与 CF 参照标量 | [reference_source_id_predictions.csv.gz](completed/reference_source_id_predictions.csv.gz)、[reference_source_id_groups.csv.gz](completed/reference_source_id_groups.csv.gz)、[reference_source_counterfactual_predictions.csv.gz](completed/reference_source_counterfactual_predictions.csv.gz)、[reference_source_counterfactual_groups.csv.gz](completed/reference_source_counterfactual_groups.csv.gz) |

冻结文件保持原样：[protocol.json](protocol.json)、[config.json](config.json)、[cpu_checks.json](cpu_checks.json)、[launch.json](launch.json)、[runtime_preflight.json](runtime_preflight.json)。协议说明见 [GEOPRR_V2_IMAGE_STRUCTURE_SOURCE_CONTROL_CN.md](../../GEOPRR_V2_IMAGE_STRUCTURE_SOURCE_CONTROL_CN.md)。原始 run、审计工作脚本及 raw bundles、完整 contracts/populations/train arrays、RGB、weights、features 与私有路径不属于本归档。

## Known Issues

- **冻结 CF condition 文本。** Public/runtime config 的 `counterfactual_gate.condition` 继承字符串 `native learning AND native primary pass; same last15 only`；顶层 checkpoint 合同与实现固定使用 `last30`。候选 CF 未执行，因此该继承诊断文本没有已观察到的数值效应。冻结 config 保持原样。
- **Provided-label 与 C3 资格。** 本次是 provided-label saved-number 比较；没有创建或重新归一化 GT。此前 C3 审计仍为 149 checks、148 PASS、1 FAIL：physical-label-triplet source contract 中没有可供核验的 `normalized_progress`/`y` 字段，raw-label derivation 的该项覆盖仍不完整。这不是已发现的标签冲突，也不能改写为完整 PASS。
- **审计范围与独立性。** 审计为 same-family、provisional，使用保存的 scalar/metadata 与新代码静态核验。未重新打开 checkpoint，未重跑 model/torch/weights/RGB/GPU、旧测试或 raw-label derivation；历史 pixel/full-weight 来源链未获新认证。
- **摘要列范围。** 核验覆盖 numeric/null 摘要及已收集 numeric/null group 列。最大误差行 ID 从保留行派生；保存的 `maximum_error_ids` 文本与冗余 `groups` 列未收集。
- **预算与结论边界。** 单 seed、重复使用的 source 开发布局与人工 raw RG-swap view，加上 30E 对 15E 的 2x 计划训练预算，不支持 causal accuracy/表示因果归因、显著性、收敛、真实域泛化或新 blind-test claim。
