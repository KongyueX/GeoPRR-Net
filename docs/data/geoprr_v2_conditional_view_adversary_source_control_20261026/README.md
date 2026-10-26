# Conditional source-view gradient control

**已完成；科学结论为 `NOT_RETAINED / source_reading_joint_failed`。** GRL（reversal）native 的 mean / tail30 / maximum 为 **3.3223066845 / 16.0971364006 / 81.1839252710%FS**，`>5%FS` 为 **38/292**；identity15 为 **2.5431218244 / 11.8009553314 / 53.7970364094%FS、16/292**，detached control 为 **2.6759538619 / 11.6840900977 / 46.8278169632%FS、25/292**。GRL 相对 identity15 与 detached control 的两套 native 联合门槛，各自五项 predicate 全部为 false。

Identity native 原准确的 102 图中，GRL 新增 11 图 `>5%FS`；control native 原准确的 92 图中新增 6 图。GRL CF 因 native joint 失败未运行，结果为 `null`、CNN/point/critic calls 为 0、无候选 CF 文件；control CF 与 identity CF 均不能替代 GRL CF。

末轮 source online critic accuracy 从 control **82.6055313%** 降为 GRL **46.5975255%**，距离 chance 的缩减为 `0.2920305676855895`。预定 operational critic effect 通过，阅读结果仍为 `NOT_RETAINED`。研究目标尚未完成，执行已因换设备暂停；保存的结束见证为 GPU compute rows 空，本轮没有下一项模型运行或 promotion，不调 α、critic width、conditioning y、labels、scaler、steps、预算或 seed 救过，也不自动启动 full-source/长训。

科学修订 `03981213603beea957989547690af73004617844` 与 runtime receipt 修订 `5e8a0da6c7ef0970068d5abcd704b29bcd636604` 已推送。六 stage（metadata、preflight、detached_control、control_cf、reversal、finalize）均 complete/exit 0；最后 stage 于 `2026-10-26T00:56:34.202434+00:00` 完成。保存的 `2026-10-26T00:57:52.503437+00:00` root completion/idle witness 显示实际 parent `440513` 及六个 child `440514/440517/440555/442404/442517/444348` 均已退出，GPU compute rows 为空。

## 阅读结果与配对

| Arm/view | Mean %FS | Acc1/2/5 counts | Median %FS | P90 %FS | P95 %FS | Tail30 %FS | Max %FS | >1/2/5 |
|---|---|---|---|---|---|---|---|---|
| identity/native | 2.5431218244 | 102/182/276 | 1.5494510531 | 4.0915115178 | 5.4043346643 | 11.8009553314 | 53.7970364094 | 190/110/16 |
| identity/CF | 2.6353762865 | 104/189/275 | 1.4384098351 | 4.1645436361 | 5.7685494423 | 12.9459618763 | 54.2472898960 | 188/103/17 |
| detached_control/native | 2.6759538619 | 92/167/267 | 1.7107360065 | 4.4575661421 | 7.0123346150 | 11.6840900977 | 46.8278169632 | 200/125/25 |
| detached_control/CF | 2.8617163082 | 100/162/265 | 1.6120150685 | 4.8130086064 | 6.8449869752 | 12.8705381726 | 50.0665783882 | 192/130/27 |
| reversal/native | 3.3223066845 | 86/166/254 | 1.7772205174 | 5.8395713568 | 7.7916396223 | 16.0971364006 | 81.1839252710 | 206/126/38 |
| reversal/CF | null | null | null | null | null | null | null | null |

已执行的五份观察各有 292/292 图、73 个完整四图 group，输入与 scalar 均有效、失败 0；合法预测端点 0/1 不作失败。尾部为 worst 30 图的 mean。Native identity 保护集为 102 图、identity CF 为 104 图；detached native 为 92 图、detached CF 为 100 图，各自独立定义。

GRL 必须相对 identity15 与 control 同时满足 mean 不大于、tail 严格小于、maximum 不大于、`>5%FS` 数不增加、各自 error≤1%FS 准确集合无新增 `>5%FS`。两套五项均失败。Control 的 tail/max 低于 identity，mean 与 `>5%FS` 数更高；这些原始取舍完整保留。

| 比较 | 图改善/恶化/不变 | Mean group 改善/恶化/不变 | 原≤1%FS → 新>5%FS | 1% 救回/新增 | 2% 救回/新增 | 5% 救回/新增 |
|---|---|---|---|---|---|---|
| control native vs identity native | 135/156/1 | 34/39/0 | 3/102 | 51/61 | 41/56 | 4/13 |
| control CF vs identity CF | 127/165/0 | 27/46/0 | 5/104 | 54/58 | 41/68 | 4/14 |
| GRL native vs identity native | 139/153/0 | 30/43/0 | 11/102 | 50/66 | 62/78 | 9/31 |
| GRL native vs control native | 142/150/0 | 37/36/0 | 6/92 | 47/53 | 70/71 | 16/29 |
| identity CF vs native | 137/152/3 | 33/40/0 | 0/102 | 24/22 | 23/16 | 4/5 |
| control CF vs native | 126/164/2 | 31/42/0 | 0/92 | 26/18 | 12/17 | 4/6 |

两 arm 均通过原 constant 0.5 learning gate：half-constant mean=`0.1250446370304666`、half-constant tail=`0.23692208052865074`（归一化误差）。Control online critic≥.65 也通过，因而允许 control CF 和 fresh GRL；这些先决条件不代表 control 阅读优于 identity。

## Online critic 与实际执行

原 RGB/CNN-GAP-point 推理路径保持，训练分支以 h1280 与既有 source y 条件化预测 original/roll 生产程序标签 Z；Z 不是 camera 角度或错误因果真值。Detached control 对 encoder 的 critic 梯度为 0，GRL 为 `-.05`；critic 自身均按相同 BCE descent 更新。

末轮 control critic 为 `4540/5496=0.8260553129548762`，GRL 为 `2561/5496=0.4659752547307133`，各自 Z counts 为 `2748/2748`。该 operational effect 没有通过阅读联合门槛，也不认证 conditional invariance 或因果成功。

| 正式 arm/范围 | Epochs | Train image presentations | Combined backward attempts | Optimizer updates | AMP skips |
|---|---:|---:|---:|---:|---:|
| Detached control | 15 | 82440 | 1290 | 1290 | 0 |
| GRL/reversal | 15 | 82440 | 1290 | 1289 | 1（epoch 1） |
| 正式合计 | — | — | 2580 | 2579 | 1 |

正式总计 CNN/point 各 `165756` image/examples、`2595` calls；critic `164880` examples、`2580` calls。Source03 与真实 Industrial/RF/external calls 均为 0。两 arm 固定 `last15`、无 intermediate validation、scheduler、best 选择或补偿 skip。

Metadata 阶段为 5496 train rows、Z 各 2748、687 parents、1374 view quartets、2748 pairs，保存的 production FP32 paired target 最大差为 0（固定 tolerance `1e-6`）；metadata 阶段 RGB/model/GPU 均为 0。

Preflight 单列：8 unique RGB、CNN/point 各 3 calls/8 examples、critic 1 call/4 examples；1 combined backward，CNN/point/critic backward 各 1、0 optimizer updates。实际 12 次 decode 操作包含额外 4 张 source-ID CPU identity re-decodes。保存 witness 为 8517906 parameters/237 tensors/468 states、critic 82113/4、原 BN77；cached encoder462 初始保留、237 parameters 在 0opt 不变、strict468 self-reload、BN train+1/eval0，h/point F16、critic/logits/BCE F32、conditioning detached。正式两 arm 均 fresh，不复用 pref 状态。

## 成本边界

| 保存范围 | 实际值 |
|---|---:|
| Whole supervisor process | 996.8495061174035 s |
| Launch receipt → supervisor finish UTC | 996.861378 s |
| Parent child lifetimes 之和 | 995.7585619427264 s |
| Child module-entry timers 之和 | 987.8305095508695 s |
| Queue last elapsed | 996.8517815582454 s |
| 两 arm training | 926.0679816864431 s |
| Critic FP32 region CUDA events，两 arm | 0.30189404548704624 s |
| Hook host，两 arm | 0.09653088822960854 s |
| Peak observed CUDA allocated / reserved | 8264907776 / 11349786624 bytes |
| Peak observed CPU memory | 5662527488 bytes |
| 当前四个 checkpoint 文件 stat 总大小 | 440096720 bytes |
| NEWCPU，warm imports 后 | 0.9023926854133606 s |
| Finalize child module entry | 1.9917696602642536 s |
| Checkpoint I/O / point-only / total GPU compute | null / null / null |
| Cold model-load-only / final screen/queue writes | null / null |

完整六阶段 timer、各阶段训练/native/CF wall、内存与训练历史见 [execution_cost.csv](completed/execution_cost.csv)、[cost_boundaries.json](completed/cost_boundaries.json)、[training_history.csv](completed/training_history.csv)。Finalize 的 `1.991769660s` 是单独 child module-entry interval，不能作为 whole-pipeline time。

## 已完成审计与公开投影

唯一 fresh、独立非作者的 standard-library saved-number/execution CPU audit 给出 **integrity `PASS`：42230 checks、0 failed**；15780 个 floating saved-value comparisons 的绝对容差为 `1e-12`，最大差为 `1.4210854715202004e-14`。科学结论维持 **`NOT_RETAINED`**。

本次仅将审计显式批准的 10 个 aggregate 文件投影到 `completed/`。4 个 CSV 保持 plain CSV，仅统一 LF，numeric strings 原样保留；其余 JSON/Markdown 原文保存。完整 metrics、Acc/分位数、三阈值、正负 pair/group aggregate、counts、cost 和资格均在这些文件中保留。本次归档不重跑或重新计算任何科学结果。

| 内容 | 文件 |
|---|---|
| 科学与 integrity 结果 | [audit_result.json](completed/audit_result.json)、[audit_checks_summary.json](completed/audit_checks_summary.json)、[audit_report.md](completed/audit_report.md) |
| 阅读指标与六套配对 aggregate | [metrics.csv](completed/metrics.csv)、[pair_comparisons.csv](completed/pair_comparisons.csv) |
| 执行、成本与训练历史 | [execution_evidence.json](completed/execution_evidence.json)、[execution_cost.csv](completed/execution_cost.csv)、[cost_boundaries.json](completed/cost_boundaries.json)、[training_history.csv](completed/training_history.csv) |
| Completed 显式清单 | [archive_allowlist.json](completed/archive_allowlist.json) |

全目录公开路径见 [public_allowlist.json](public_allowlist.json)。冻结的 [protocol.json](protocol.json)、[config.json](config.json)、[cpu_checks.json](cpu_checks.json)、[launch.json](launch.json)、[runtime_preflight.json](runtime_preflight.json) 与 [中文协议](../../GEOPRR_V2_CONDITIONAL_VIEW_ADVERSARY_SOURCE_CONTROL_CN.md) 保持原样。逐行/逐组 IDs、targets、source arrays、完整 contracts、features、RGB、weights、私有逐项比较与 verifier 不在公开包中。

## Limitations

- **Provided-value 与 raw GT。** Production `normalized_target` 到 FP32 consumer 的绑定与 paired difference 0 是保存的 value witness，没有重新推导或认证 physical GT。C3 原 149/148/1 raw-derivation 资格仍未解决，不升级为 PASS。
- **Metadata 读取范围。** `prepared` metadata 的 nested train/ID/OOD contract 被私有 parse/copy，只有 training fields 驱动 y/Z；metadata 阶段没有打开 ID/OOD manifest 或 pixels，不能把完整 metadata parse 称为 strict train-only 文件读取。
- **预算与实际 updates。** 两 arm scheduled 15E/1290 attempts 匹配，effective updates 为 1290 对 1289；GRL epoch 1 有一次正常 AMP skip，没有补偿 steps 或调整 scaler。
- **Online critic。** 每 epoch online source critic counts 包含全部 5496 attempted presentations（含 skip）；这是 co-trained 统计，不是独立 held-out/frozen-feature probe，不证明 conditional mutual information 为 0、conditional invariance 或因果成功。
- **轨迹与初始化。** 两 arm joint gradient clipping 都包含 critic；detached control 不等于旧 identity 的 exact weight trajectory。Fresh seed/order、forked RNG 代码与保存 initialization metadata 支持相同形式初始化，但没有新实测 cross-arm 初始 tensor 相等。
- **Pref 与历史来源链。** Pref strict self-reload、cached462/237-at-0opt/468-state/BN77 witnesses 是保存证据；没有重新读取历史 pixel/full weights，不构成其新的完整认证。
- **FP32 forward sum。** 每 batch 的 forward numerical sum 在 FP32 计算，其保存平均值与分别平均的 primitive/BCE 再相加可有 `>1e-12` residual；control 最大 `3.1884580531382767e-09`、GRL 最大 `7.298749271633653e-09` 原样保留，不改写数值。
- **计时与内存。** Child module timer 始于 torch import 前但不含最终 JSON/down-time；supervisor 包含 child lifetime 至 finalize，未覆盖最终 screen/queue writes 与 caller lifetime。CUDA event region 包含 concat、critic、primitive、BCE、forward sum，不是纯 MLP 用时；hook host 单列。CPU peak 是单 stage `RUSAGE_SELF` RSS 的最大观察值，不含 loader workers 或 whole-host memory。Phase/forward wall 不是 total GPU compute。
- **未测成本。** Checkpoint I/O、point-only、GPU total、cold-load-only、screen/final writes 均为 `null`。四 checkpoint 仅 stat，没有读取 weights bytes 或 state。
- **结束见证。** Readiness 采用保存的 `00:57:52.503437UTC` root monitor；final child 退出前写下的 parent-exit flags 仍为 false，并非其后 liveness 故障。本次审计与归档没有新 process/GPU 查询。
- **未运行范围与泛化。** GRL CF 为 null/0 calls，不能以 control/identity CF 代替；source03/真实 Industrial/RF calls 为 0。单 seed、已查看 source dev 与人工 CF 不支持显著性、新 blind test、收敛或真实域泛化，也不触发 promotion。
- **公开包可重跑范围。** 公开文件报告这一次已完成审计；原 verifier 依赖私有 validation rows/run context 且不公开，不能声称本公开包可独立重跑。

设备迁移更新：真人因旧服务器费用不足要求暂停自动续跑并换设备；heartbeat现为PAUSED，不在旧机开新任务。见[当前进度](../../GEOPRR_CURRENT_PROGRESS_CN.md)和[迁移交接](../../GEOPRR_DEVICE_MIGRATION_HANDOFF_CN.md)。这不修改上述封存实验结果或冻结协议。
