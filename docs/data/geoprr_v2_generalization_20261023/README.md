# MobileNetV4泛化迭代：完整短训练结果与已有模型定位

2026-10-23。**真实端点源覆盖的匹配收益已在两个原生读出家族得到支持。v12 local端点best1的Industrial／RF100为20.539125／7.659157%FS，相对同local57初始化和相同四轮预算的原源控制分别降低12.6263%／19.8292%；预先固定的last4比较也降低6.2913%／10.8692%。** 相对原local57，best1两域降低6.3916%／16.3796%，但源1506均值0.083392%FS上升8.1298%；3172由5.908847恶化至18.677585%FS，是原有>5%FS成员，未新增该阈值成员。完整best／last、普通与严重错误、零位／非零、逐图／组效应和真实成本见[v12完整结果](v12_completed/report_cn.md)、[完整指标](v12_completed/metrics.json)、[303124项数值与执行核对](v12_completed/numeric_execution_audit.json)与[28文件公开清单](v12_completed/public_allowlist.json)。

v11 feedback端点best1的Industrial17.353598%FS仍更低，v12 local端点best1的RF1007.659157%FS更低；保留各自源尾部代价，不将一个模型称为所有数据集最优。v11相对匹配正常BN控制两域均值降低4.7510%／5.3492%，源均值相对feedback48上升7.6628%，2810新增>5%FS。其固定BN臂源均值更低，但两域均差于匹配固定BN控制。见[v11完整结果](v11_completed/report_cn.md)与[完整指标](v11_completed/metrics.json)。端点数据带来的开发增量不自动构成算法创新认证。

v9固定BN best1的局部收益保留：RF100为9.473928%FS、Industrial为18.709661%FS；MixStyle未取得一致增量。v10 ROI留边两臂均使源域及两外域回退，不保留、不延长。所有旧模型和负结果均保留，分别见[v9完整结果](v9_completed/report_cn.md)与[v10完整结果](v10_completed/report_cn.md)。

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

## 已完成的v9首批训练合同

共同初始化选择v6 feedback best48，理由是它在已查看的两套外部开发人口上均优于v7，同时保留local57在RF100上的优势作为比较。外部开发选锚点这一事实明确记录；不是源域盲选或新盲测。

三臂各4轮、seed20262020、fresh optimizer，均从同一完整496-state／7,461,060参数权重开始；源18,590 unique、每轮22,686 donor＋8,192 recipient／709批，原三项loss及768／192不变。共享LR1e-5、state head5e-5，无warmup。仅比较：

1. 低LR续训参照，BN正常更新。
2. 固定backbone BN运行统计，affine仍训练。
3. 配方2加早期`blocks.0`／`blocks.1`的训练期MixStyle（stride4／8，p0.5／alpha0.1）。

两项干预均无新增参数；MixStyle在eval时严格identity。CPU定向5项检查与实际RGB／反传检查通过：2张unique RGB、5次CNN调用／10图像前向、3次反传、0optimizer更新；三臂梯度有限，固定BN两臂228个buffers保持不变，参照发生预期变化。这个小batch witness不是正式成本或性能结果。

每臂完整4轮，best-source分别为control第1轮、bnfixed第1轮、MixStyle第3轮；末轮均为第4轮。两种checkpoint分别完成两个外部开发人口评估，合计12份外评、9,276张CNN图像／588次调用、0反传／0更新。一次独立CPU核对35,216项比较全部通过。评价连续误差、普通／近零／尾部和实际成本，允许局部回退，不以原零新增错门槛一票否决。

v10几何一致的源ROI留边实验现已完整结束：两臂各4轮、8份外评，网络和loss不变，复用v9对应控制。一次独立CPU核对80,618项比较全部通过，性能回退按原值保留。见[ROI协议](../../GEOPRR_V2_ROI_GENERALIZATION_PROTOCOL_CN.md)、[v10完整结果](v10_completed/report_cn.md)、[输入／运行见证](v10_runtime_preflight.json)和[启动记录](v10_launch.json)。

v11将首128个quartile源组重新渲染为真实RGB状态[0,0.005,0.02,1]，保持18590人口与原曝光预算。128组／512张真实RGB与物理标签检查、0更新运行见证、两臂各4轮及6份实际外评均完整结束；固定BN best4与last的外评复用。一次独立CPU核对245849项全部通过，见[完整结果](v11_completed/report_cn.md)、[源阶段记录](v11_source_completed.json)、[运行见证](v11_runtime_preflight.json)和[外评别名记录](v11_completed/external_aliases.json)。

v12原生v5 local57的正常BN原源控制与端点源两臂均完整4轮、best均为第1轮；8份实际外评共6184张CNN图像／392次调用，无别名或重复推理。一次独立CPU数值与执行核对303124项全部通过。输入metadata确认8192条实际训练配对在512-ID映射下顺序精确保留；保存的data_contract.edges按新ID规范排序，与loader采样顺序不同，不能据此推断实际训练配对顺序改变。见[完整结果](v12_completed/report_cn.md)、[local端点协议](../../GEOPRR_V2_LOCAL_ENDPOINT_GENERALIZATION_PROTOCOL_CN.md)和[运行见证](v12_runtime_preflight.json)。

下一独立v13计划在feedback48与local57两个原生家族中仅固定CNN权重，源BN运行统计继续更新，decoder和head继续训练；使用同一端点源，各4轮，匹配控制复用v11／v12正常BN全量微调结果。v13尚未启动，协议／代码和新运行见证正在准备；固定已特化CNN权重不等于恢复ImageNet通用表征。

## 文献依据

- [MixStyle，ICLR 2021](https://arxiv.org/abs/2104.02008)：训练时混合早期实例通道统计，保留空间位置，不混角色标签。此处是其低层原则在MobileNet中的具体实现，采用已有工具不自动构成新创新。
- [Schneider等，NeurIPS 2020](https://arxiv.org/abs/2006.16971)：BN统计与分布偏移相关；原文目标统计适配的收益不等于固定源统计必然有效。本轮固定统计是待证伪对照，不做目标适配。
- [Kumar等，ICLR 2022](https://arxiv.org/abs/2202.10054)：微调可能扭曲预训练特征。冻结已特化feedback48不等于恢复ImageNet表征；本轮不作这种归因。

## Limitations

Industrial与RF100已经用于开发观察；结果不称新盲测。各已有版本训练历史不同，本表不能单因素归因结构。短训练是单seed开发筛选，不承诺外域收益、显著性或论文录用。v9归一化／风格和v10 ROI干预未补充缺失零位；v11／v12真实端点覆盖取得匹配跨域正结果，同时保留源严重错误代价，未证明所有零位或背景偏移已经解决。原14494／1506／4000与原门槛失败、旧模型和负结果均保留。RGB、权重、run.json原始几何与private trace不进入公开包。

执行细节见[泛化协议](../../GEOPRR_V2_GENERALIZATION_ITERATION_PROTOCOL_CN.md)。本包保存family逐图与标量、[v10完整结果](v10_completed/report_cn.md)，以及[v9完整结果](v9_completed/report_cn.md)、[已核对的完整指标](v9_completed/metrics.json)、[数值与执行核对](v9_completed/numeric_execution_audit.json)和[公开文件清单](v9_completed/public_allowlist.json)。

端点数据配方与训练边界见[端点覆盖协议](../../GEOPRR_V2_ENDPOINT_GENERALIZATION_PROTOCOL_CN.md)。
