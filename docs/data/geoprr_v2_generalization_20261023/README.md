# MobileNetV4泛化迭代：完整短训练结果与已有模型定位

2026-10-23。首批三个4轮训练臂均已完整收尾，没有一臂相对初始化feedback48同时改善Industrial与RF100。固定BN best1在RF100的NMAE降至9.473928%FS（相对下降13.39%），Industrial升至18.709661%FS（相对上升4.75%）；MixStyle未取得一致跨域增量。保留全部模型与负结果，不直接延长本轮训练。完整比较、普通与严重样本统计、成本和公开文件清单见[v9完整结果](v9_completed/report_cn.md)。

相同的Industrial1395与RF100151原生ROI比较中，已有角度模型feedback48的Industrial均值最低，local57的RF100均值最低；v7／v8的源域精度优势不等于跨域优势。

| 已有模型 | Industrial NMAE %FS | RF100 NMAE %FS |
|---|---:|---:|
| 初始field best61 | 25.688460 | 10.884443 |
| v3 supervised best63 | 30.822418 | 10.009492 |
| v5 local joint best57 | 21.941540 | **9.159439** |
| v6 feedback best48 | **17.862103** | 10.938932 |
| v7 full best64（复用） | 21.226570 | 12.691067 |
| v8 global best2 | 21.571843 | 12.854921 |

10份新评估均正常退出，共7730张CNN图像前向／490次调用，0反传／0更新；v7复用既有同人口结果，没有重复推理。逐图与组／子集统计的一次独立CPU核对通过。每张ROI、FP32 target、group、component与既有v7人口一致，失败样本保留全分母，未改变各模型的原生读出。

## 已完成的首批训练合同

共同初始化选择v6 feedback best48，理由是它在已查看的两套外部开发人口上均优于v7，同时保留local57在RF100上的优势作为比较。外部开发选锚点这一事实明确记录；不是源域盲选或新盲测。

三臂各4轮、seed20262020、fresh optimizer，均从同一完整496-state／7,461,060参数权重开始；源18,590 unique、每轮22,686 donor＋8,192 recipient／709批，原三项loss及768／192不变。共享LR1e-5、state head5e-5，无warmup。仅比较：

1. 低LR续训参照，BN正常更新。
2. 固定backbone BN运行统计，affine仍训练。
3. 配方2加早期`blocks.0`／`blocks.1`的训练期MixStyle（stride4／8，p0.5／alpha0.1）。

两项干预均无新增参数；MixStyle在eval时严格identity。CPU定向5项检查与实际RGB／反传检查通过：2张unique RGB、5次CNN调用／10图像前向、3次反传、0optimizer更新；三臂梯度有限，固定BN两臂228个buffers保持不变，参照发生预期变化。这个小batch witness不是正式成本或性能结果。

每臂完整4轮，best-source分别为control第1轮、bnfixed第1轮、MixStyle第3轮；末轮均为第4轮。两种checkpoint分别完成两个外部开发人口评估，合计12份外评、9,276张CNN图像／588次调用、0反传／0更新。一次独立CPU核对35,216项比较全部通过。评价连续误差、普通／近零／尾部和实际成本，允许局部回退，不以原零新增错门槛一票否决。

下一轮几何一致的源ROI留边与偏移实验已于北京时间2026-10-23 03:08:26启动：两臂各4轮，网络和loss不变，复用本轮对应控制；实际父子进程和唯一GPU已核实，首批训练loss有限。目前尚无完整性能结果。见[ROI协议](../../GEOPRR_V2_ROI_GENERALIZATION_PROTOCOL_CN.md)、[输入／运行见证](v10_runtime_preflight.json)和[启动记录](v10_launch.json)。

## 文献依据

- [MixStyle，ICLR 2021](https://arxiv.org/abs/2104.02008)：训练时混合早期实例通道统计，保留空间位置，不混角色标签。此处是其低层原则在MobileNet中的具体实现，采用已有工具不自动构成新创新。
- [Schneider等，NeurIPS 2020](https://arxiv.org/abs/2006.16971)：BN统计与分布偏移相关；原文目标统计适配的收益不等于固定源统计必然有效。本轮固定统计是待证伪对照，不做目标适配。
- [Kumar等，ICLR 2022](https://arxiv.org/abs/2202.10054)：微调可能扭曲预训练特征。冻结已特化feedback48不等于恢复ImageNet表征；本轮不作这种归因。

## Limitations

Industrial与RF100已经用于开发观察；结果不称新盲测。各已有版本训练历史不同，本表不能单因素归因结构。短训练是单seed开发筛选，不承诺外域收益、显著性或论文录用。归一化／风格干预不能补造缺失零位，端点与几何一致的ROI覆盖仍是后续候选。原14494／1506／4000与原门槛失败、旧模型和负结果均保留。RGB、权重、run.json原始几何与private trace不进入公开包。

执行细节见[泛化协议](../../GEOPRR_V2_GENERALIZATION_ITERATION_PROTOCOL_CN.md)。本包保存family逐图与标量，以及[v9完整结果](v9_completed/report_cn.md)、[已核对的完整指标](v9_completed/metrics.json)、[数值与执行核对](v9_completed/numeric_execution_audit.json)和[公开文件清单](v9_completed/public_allowlist.json)。
