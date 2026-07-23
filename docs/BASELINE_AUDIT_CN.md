# 同类方法可复现性审计（截至 2026-07-23）

本文必须区分“内部消融”和“外部同类方法”。Geometry-v1/v2、融合、残差及门控
都属于本文内部变体，不能代替外部模型对比。下面记录提交论文前实际核验过的公开资源，
避免把论文中声称“将发布”的代码或权重误写成已复现结果。

| 方法 | 核验提交/来源 | 代码与权重状态 | 当前处理 |
|---|---|---|---|
| VDN / Pointer-10K | `DrawZeroPoint/VectorDetectionNetwork@68afe1e` | GPL-3.0；有训练、推理和数据格式，README 权重链接为空 | 已在 SyncG train 完成重训；写作 `VDN architecture, retrained on SyncG` |
| Learning to Read Analog Gauges from Synthetic Data (WACV 2024) | `fuankarion/automatic-gauge-reading@a7d5956` | 声明的仓库仍近乎为空，无可执行实现和权重 | 只放相关工作，不伪造复现成绩 |
| Human-like Alignment and Reading (2023) | `shuyansy/Detect-and-read-meters@e5e1680` | MIT；有训练/测试代码、数据和检测/读数/对齐权重；`v2` 移除了论文中的耗时 STN 对齐模块 | 原论文 MC1296 协议不同；可作下一项复现，不把论文原数值混入 SyncG 主表 |
| Human-like Keypoint Sequence (Measurement 2025) | `paopao6777/det-read-pointer-meter@a79bdea` | 有代码说明；仓库无许可证，README 说明数据不能公开 | 只列论文原协议；当前无法做同数据重训 |
| TransUNet meter reading (2024) | 期刊官方页面 | 自采 Simple/Complex 数据不公开 | 只列相关工作及其原协议结果 |
| DialBench / MRLM | `Event-AHU/DialBench@707bcdc` | LICENSE 为 MIT、README 却称 BSD-3-Clause；2026-07-21 公告新增百度模型权重链接，但 Model Zoo 仍标 `TBD`，数据/模型许可说明仍不完整 | 单独报告完整 RPM-10K VLM 协议；不能与本文已知量程的 1,797 张子集直接排名 |

公开入口：

- [VDN 官方仓库](https://github.com/DrawZeroPoint/VectorDetectionNetwork)
- [WACV 2024 方法声明的官方仓库](https://github.com/fuankarion/automatic-gauge-reading)
- [Human-like Alignment and Reading 官方仓库](https://github.com/shuyansy/Detect-and-read-meters)
- [Human-like Keypoint Sequence 官方仓库](https://github.com/paopao6777/det-read-pointer-meter)
- [TransUNet 表计读数论文](https://www.mdpi.com/2079-9292/13/13/2436)
- [DialBench 官方仓库](https://github.com/Event-AHU/DialBench)

## 最小可发表比较方案

正文主对比保留以下四类即可：

1. 传统几何读数（无学习）；
2. 项目原始 Transformer；
3. VDN 的公开架构重训版；
4. 本文完整方法。

其余内部方法移到消融表。VDN输出的是指针向量，不直接输出标量读数；公平接入时应让它与
本文共享同一个表盘检测、起终刻度和已知量程适配器，并同时报告其原生方向角误差和适配后的
NMAE/Acc@2%。必须注明共享组件，不能写成未经修改的官方端到端成绩。

## VDN 实际复现审计

- 外部源码固定为完整提交 `68afe1efbdb35d3196d9a6243bfac8e5c9de5ceb`，运行前同时检查
  Git 提交、tracked worktree 洁净状态和 GPL-3.0 声明；外部源码位于 Git 忽略目录，未复制进
  本项目源码。
- 使用官方 ResNet-18、三层反卷积、热图头和二维向量头，输入/输出为 384/96；SyncG 的
  `outside_kp`/`origin_kp` 分别映射为针尖/针尾，表盘框采用官方 1.25 倍正方形仿射裁剪。
- 正式子集均为单指针，推理使用官方同仓库 `get_max_preds` 对应的热图全局最大值，再在该点
  采样向量并归一化；这避免把官方为多指针场景返回的多个局部峰任意混入一个标量读数。
- 本地适配器的仿射矩阵、热图和向量监督已与固定提交中的官方函数逐元素比对：监督张量最大
  差值为 0，仿射矩阵误差小于 `1e-5`。
- 初始化文件严格使用官方配置命名的 `resnet18-5c106cde.pth`，SHA-256 为
  `5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8`。
- 官方 YAML 虽写有 `WD=0.0001`，但固定提交的 `get_optimizer` 调用 Adam 时没有传入
  `weight_decay`。因此忠实复现采用 Adam、初始学习率 `1e-3`、有效 weight decay 为 0；代码
  测试会直接解析官方函数并验证这一事实。误按 YAML 施加 weight decay 的诊断运行已隔离，
  不进入任何正式统计。
- 官方 200 epoch 的 `140/190` 降学习率位置按比例压缩为预声明的 100 epoch、`70/95`；
  batch size 为 8，旋转/尺度增强和随 epoch 线性增加的向量损失权重保持官方定义。训练/验证按
  `meter type + scene` 的 725 个组划分，验证组不与训练组重叠。
- 正式运行完成后，验证器必须逐轮核对学习率、向量损失权重、样本数、优化器步数、有限损失、
  最佳 epoch、源码哈希和主干权重统计；任何一项不一致都禁止启动测试集评测。

复现已完成。100 轮训练的最佳检查点来自第 99 轮，验证方向 MAE 为 `0.666765°`；
14,375/1,625 个训练/验证样本对应 652/73 个零交集组。179,644 个有效优化步与 56 个
AMP 跳步之和等于预期的 179,700 个批次。检查点、训练摘要和评估源码哈希均通过独立
验证器，随后才运行 SyncG clean、五种控制退化与 RPM 共七组冻结评测。

| 条件 | VDN NMAE | Ours NMAE | Ours−VDN 95% CI | 结论 |
|---|---:|---:|---:|---|
| Clean | 0.1482 | 0.1152 | [-0.0362, -0.0299] | Ours 显著更好 |
| Moderate / severe blur | 0.1481 / 0.1554 | 0.1152 / 0.1424 | 均完全小于 0 | Ours 显著更好 |
| Moderate perspective | 0.1486 | 0.1381 | [-0.0137, -0.0072] | Ours NMAE 显著更好 |
| Severe perspective | 0.1712 | 0.1707 | [-0.0040, +0.0029] | 统计持平；VDN Acc@2% 更高 |
| Severe blur + perspective | 0.2364 | 0.2526 | [+0.0115, +0.0211] | VDN 显著更好 |
| RPM single-pointer | 0.3813 | 0.4973 | [+0.0168, +0.2105] | VDN 显著更好 |

完整自动汇总位于
`artifacts/runs/vdn_syncg/seed_20260720/vdn_comparison.{json,md}`。因此这项外部对比
支持 clean、模糊和中度透视上的有限优势，但不支持“异常视角全面领先”或 SOTA。RPM 上
VDN 的端到端 coverage 为 98.78%，原掩码分支仅为 72.73%。这一诊断直接驱动了下节的
分割无关方向回退实验，而不是继续用 RPM 标签微调残差回归器。

## 第一代独立方向头、硬回退与质量路由审计（历史消融）

上述失败分析之后，本文新增了本地独立实现的支点—方向头。它采用 torchvision ResNet-18
编码器、三层上采样热图解码器和全局二维单位向量头，输入/热图尺寸为 256/64。实现只复用
本项目通用的 SyncG 清单与裁剪辅助函数，不导入、复制或运行 VDN 的 GPL-3.0 源码；因此该头
属于本文方法，VDN 继续作为外部基线，两者不能在写作中混同。

主种子 `20260722` 只在 SyncG train 上训练 30 epochs，14,398/1,602 个训练/验证样本来自
653/72 个零交集场景组。验证器核对 9,000 个优化步、0 次 AMP 跳步、有限模型状态、训练
清单/样本 ID/源码/初始化权重/检查点哈希；最佳检查点 SHA-256 为
`90524dec4ef27cb8c6656f7a4055967c711107f48586e5d6f76b99f47907faaf`。ImageNet 初始化文件
`resnet18-f37072fd.pth` 的 SHA-256 为
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

最终硬路由只在原掩码分支无输出时接管，不设置目标域置信度阈值。RPM 标签从未参与训练、
早停、路由或模型选择。冻结评测结果为：

| 方法 | RPM NMAE ↓ | RPM Acc@2% ↑ | RPM coverage ↑ |
|---|---:|---:|---:|
| Ours-mask | 0.4973 | 0.0401 | 0.7273 |
| Independent direction | 0.3433 | 0.0568 | 0.9878 |
| Ours-hard dual route | **0.3346** | 0.0484 | 0.9878 |
| VDN retrained | 0.3813 | **0.0595** | 0.9878 |

Ours-hard 相对 VDN 的配对 `ΔNMAE=-0.0467`，分组 bootstrap 95% CI 为
`[-0.0889, -0.0199]`；但 Acc@2% 较低。因此可以主张 RPM 子集上的 NMAE 与覆盖率结果，
不能主张全部指标领先、完整 DialBench 排名或 SOTA。严重组合退化上 Ours-hard NMAE 为
0.2510，仍显著差于 VDN 的 0.2364，也必须保留为负面结果。

三个独立完整训练种子的 RPM 最终 NMAE 为 `0.3317 ± 0.0039`，相对固定 VDN 的差为
`-0.0496 ± 0.0039`；三次均恢复 468/490 个硬失败。训练侧验证方向 MAE 为
`1.7099 ± 0.1726°`。该稳定性结果排除了主结论只来自单次初始化的可能，但显著性判断仍使用
逐样本分组 bootstrap，而不是把三个种子当作三个数据样本做显著性检验。

随后只用 SyncG train 构造了 4,380 张、197 组的跨模型 OOF 路由集：掩码读数来自 grouped
OOF 残差模型，方向读数来自从未见过该仪表组的方向检查点。正式验证器确认组泄漏数、测试
样本使用数和正式测试样本 ID 交集均为 0。运行时 ExtraTrees 路由器及阈值完全由该集合拟合；
RPM、SyncG test 和五种退化不参与特征或阈值选择。

第一代 `quality route-v1` 的结果为：

| 方法 | Clean NMAE ↓ | Combined-severe NMAE ↓ | RPM NMAE ↓ | RPM Acc@2% ↑ | RPM coverage ↑ |
|---|---:|---:|---:|---:|---:|
| Ours-hard | 0.1146 | 0.2510 | 0.3346 | 0.0484 | 0.9878 |
| Quality route-v1 | **0.1071** | **0.2316** | **0.3241** | 0.0562 | 0.9878 |
| VDN retrained | 0.1482 | 0.2364 | 0.3813 | **0.0595** | 0.9878 |

六个 SyncG 条件中，Quality route-v1 相对 Ours-hard 的配对区间都低于 0；相对 VDN 也全部低于
0。RPM 上相对 VDN 的 `ΔNMAE=-0.0572`，95% CI `[-0.1101,-0.0206]`；但相对 Ours-hard
的 `ΔNMAE=-0.0105` 区间 `[-0.0294,+0.0063]` 跨 0。该结果现在保留为关键历史消融，
不能再称为最终方法，也不能写成它在 RPM 上显著优于 hard route。

第一代质量路由的模型哈希为
`679200b26ca414c53504002c83434c7655830aac383a1dda428ee283b422260d`。七组输出已逐行重算
route、NMAE、Acc@2% 和 coverage，0 个决策或预测不一致。还需披露：该路由虽没有标签/阈值
泄漏，但方法构想受前一轮 test 失败分析启发，故不是严格意义上的全流程盲测。

## 当前主对比：概率方向 + 进度校准路由 vs VDN

当前最终方法在第一代方向头上增加概率圆周方向、精确单应投影配对和等变约束，再只用
SyncG train grouped-OOF 拟合 angle-to-progress 校准器与 mask/vector 安全路由。VDN 与本文
方法共享输入、表盘框、起终参考、已知量程适配器、退化样本和失败惩罚，因此下表是当前唯一
可以放进同一数值主表的外部架构比较：

| 条件 | VDN NMAE | Ours-final NMAE | Ours−VDN 95% CI | VDN / Ours Acc@2% |
|---|---:|---:|---:|---:|
| Clean | 0.1482 | **0.0890** | [-0.0626, -0.0560] | 0.3058 / **0.4818** |
| Moderate blur | 0.1481 | **0.0905** | [-0.0608, -0.0544] | 0.3080 / **0.4810** |
| Severe blur | 0.1554 | **0.1082** | [-0.0503, -0.0442] | 0.2730 / **0.4088** |
| Moderate perspective | 0.1486 | **0.1149** | [-0.0370, -0.0303] | 0.2823 / **0.3190** |
| Severe perspective | 0.1712 | **0.1454** | [-0.0290, -0.0227] | **0.1873** / 0.1848 |
| Severe blur + perspective | 0.2364 | **0.2184** | [-0.0219, -0.0143] | 0.1525 / **0.1708** |
| RPM single-pointer | 0.3813 | **0.2842** | [-0.1154, -0.0712] | **0.0595** / 0.0551 |

因此可以主张“在统一协议的七个条件上，最终方法 NMAE 显著低于重训 VDN”。不能扩写为
“全部指标全面领先”：severe perspective 与 RPM 的 Acc@2% 仍由 VDN 略高。自动生成的
机器可读结果位于
`artifacts/runs/calibrated_progress_router_syncg/calibrated_progress_comparison.{json,md}`。

## 跨协议论文如何比较

HARR、WACV 2024、TransUNet、2025 keypoint sequence 和 DialBench/MRLM 的输入、量程信息、
数据划分、失败处理和指标均与本文不同。论文中应另设“原论文协议与资源状态”表，引用其原文
数值但标注 `not directly comparable`，不能与上表按大小排序。例如 DialBench 的完整
RPM-10K 任务还要求从原图推断量程/刻度文本，而本文 RPM 子集把官方 range 当作已知元数据；
这两个任务不相同。

为避免“没有提到现有方法”，相关工作表可以列原论文数值，但必须保留协议栏：

| 论文方法与原协议 | 原论文报告 | 为什么不能与本文主表直接排序 |
|---|---|---|
| [Human-like Alignment and Reading](https://arxiv.org/abs/2302.14323)，MC1296 | Ref `0.26%`、Rel `1.70%`、约 `25 FPS` | 专用 MC1296、对齐/STN 流程和成功样本指标不同 |
| [Human-like Keypoint Sequence](https://www.sciencedirect.com/science/article/pii/S0263224125003537)，自建实验室/真实集 | 示值误差 `0.039% / 0.733%`，CPU `3.61 FPS` | 数据不可公开，示值误差不是本文全分母 NMAE |
| [TransUNet meter reading](https://www.mdpi.com/2079-9292/13/13/2436)，Simple/Complex | accuracy `97.81% / 93.39%` | 自采数据不公开，accuracy 容差定义与 Acc@2% 不同 |
| [DialBench / MRLM](https://arxiv.org/abs/2511.21982)，完整 RPM-10K | Accε `62.4%`、Accθ `70.9%`、Ref `0.063`、Rel `0.535` | 8.7B VLM、完整 2,000 张协议并同时识别量程/文本 |
| [Learning from Synthetic Data](https://openaccess.thecvf.com/content/WACV2024/html/Leon-Alcazar_Learning_to_Read_Analog_Gauges_from_Synthetic_Data_WACV_2024_paper.html) | 论文报告平均误差减少 `4.55`（相对 `52%`） | 4,813 张自建真实图、实现和权重当前不可得 |

这些数值用于定位工作，不是本文的复现实验结果。正文的可执行外部数值主表仍以 VDN 为准；
若投稿审稿人要求第二个同协议可执行基线，优先复现 HARR，但必须固定其版本、重新适配
SyncG/RPM，并在运行前预声明训练轮数和量程转换规则。

## 数据和训练约束

- Pointer-10K 官方入口目前是百度网盘，许可为 CC BY-NC-SA 4.0；Hugging Face Hub
  检索未发现可核验的同名镜像。若使用第三方镜像，必须先用官方文件清单或哈希核验内容。
- VDN 用 SyncG train 重训，主表应写 `VDN architecture, retrained on SyncG`，并与本文使用
  完全相同的SyncG train/test以及控制退化图像。
- 若VDN用Pointer-10K重训，则写 `VDN, retrained on Pointer-10K`，结果属于跨域迁移设置；
  不可和同域SyncG训练方法含混表述。
- 当前已完成的 VDN 对比仍不足以支持 SOTA；其他方法因无公开权重、无可执行代码或评测
  协议不同，只能列入相关工作与可复现性限制，不能填入推测成绩。
