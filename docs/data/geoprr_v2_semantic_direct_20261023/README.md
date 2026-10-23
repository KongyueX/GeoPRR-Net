# MobileNetV4完整语义直接读数：60轮完整结果

2026-10-23。**v16完整语义模型的Industrial NMAE为11.686054%FS，已接近旧GeoPRR历史11.643618%FS但仍更高；RF100为10.164776%FS，差于旧历史6.607079%FS和v12 local端点版7.659157%FS。** 源验证best59为0.526217%FS、max65.953931%FS，>1／2／5%FS为127／40／13。60轮和冻结best59的两套外评均已完成，没有全面最优的新模型。

| 保存模型／checkpoint | 源1506 NMAE %FS | Industrial1395 NMAE %FS | RF100151 NMAE %FS |
|---|---:|---:|---:|
| 旧GeoPRR clean历史 | — | 11.643618 | 6.607079 |
| 原生feedback48几何，同v15前向 | 0.063613 | 17.862102 | 10.938938 |
| v11 feedback端点best1 | 0.068488 | 17.353598 | 9.643770 |
| v12 local端点best1 | 0.083392 | 20.539125 | 7.659157 |
| v16完整语义best59 | 0.526217 | 11.686054 | 10.164776 |
| v16 last60，未另作外评 | 0.541737 | — | — |

相对v11／v12端点版，Industrial均值分别下降32.6592%／43.1034%；RF100分别回退0.521006／2.505619个百分点。v16使v15冻结GAP小头探针的源误差17.178667%FS降到0.526217%FS，但源几何精度和RF100仍有明显代价。全部候选和负结果保留，独立[v17空间残差读出协议](../../GEOPRR_V2_SEMANTIC_SPATIAL_ZERO_SHOT_PROTOCOL_CN.md)已冻结，源训练与结果以独立记录为准。

[完整报告](completed/report_cn.md)、[完整指标](completed/metrics.json)、[60轮history](completed/history.jsonl)、[LF比较表](completed/comparison.csv)、[81479项数值／执行核对](completed/numeric_execution_audit.json)和[18文件完成清单](completed/public_allowlist.json)已归档。源best59／last60各1506条预测，外域1395／151条预测与52／35组CSV、summary完整保存；普通1495／1497、严重6、3172／2810／3663／9323、新max8195、Acc、zero／nonzero、tail／max及所有救回／新增ID均保留。

## 模型、输入与选择

完整MobileNetV4使用已有ImageNet缓存，保留分类语义通路conv_head960→1280和norm_head，encoder462 state／8434512参数；新增1281参数point head后总8435793参数／464 state，实际embedding1280。不是从旧任务checkpoint接线性小头。源18,590 unique和原验证1506，require_geometry=False，模型和SmoothL1(beta0.05)仅使用RGB及源归一化scalar。原bbox只作既有ROI准备，不计算或输入H／field／mask／虚拟角色。

声明256 RGB、LINEAR resize、ImageNet归一化，源训练使用原光度增强。全模型60完整epoch、batch64／val64、workers4、seed20262020、AMP、fresh AdamW；前30轮cosine3e-4，后30轮cosine3e-5。source1506 NMAE选择最早best59，last60源表保留；Industrial／RF100没有用于训练、统计量、BN适配、checkpoint或超参数选择，source-best冻结后每域一次外评。

## 实际执行与成本

完整17460次源训练CNN调用／1115400张图、源验证1440次调用／90360张图，外部25次调用／1546张图，正式合计1207306张CNN图像／18925次调用。源反传17460次、有效更新17454次、AMP skip6；目标反传／更新均0，无resume。核心训练2761.025849秒、源验证183.317439秒，完整训练process2973.163637秒；Industrial／RF100完整process11.406062／10.302606秒，controller整体2997.258767秒。新256输入见证1次CNN／4张源图／1次反传／0optimizer单列，不计正式成本。

新语义维度、原ImageNet全encoder加载和源输入检查见保留的[4项CPU检查](cpu_checks.json)、[运行见证](runtime_preflight.json)和[启动记录](launch.json)；初始化／完成标量见[初始化记录](completed/initialization_record.json)、[完成记录](completed/completed_record.json)。完成归档复用一次独立CPU核对81479项PASS、0失败、最大数值差1.42e-14，没有重新加载模型、使用GPU或重复旧GT／测试。

## 旧clean优势与当前错误模式

旧GeoPRR clean全部relation_invalid，Raw base与relational一致；base的Industrial／RF100已为11.642999／6.569962%FS，最终为11.643618／6.607079%FS。该优势不能直接归给SARN、polar或router。当前模型在RF100既救回部分几何大错，也产生新的低值到高值偏移：046的target0.135899被预测成0.680029，误差54.413070%FS；072／092则分别从v12 local的40.639678／38.764992降至0.133330／0.122091%FS。报告保留了相反方向案例、全量paired和分组结果。

本阶段已经完整结束，独立[v17空间残差读出协议](../../GEOPRR_V2_SEMANTIC_SPATIAL_ZERO_SHOT_PROTOCOL_CN.md)已冻结，源训练与结果以独立记录为准；目前不将描述性互补当成已实现融合，也不直接延长60轮。正式配方与历史预算见[完整语义协议](../../GEOPRR_V2_FULL_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。

## Limitations

- 本轮同时改变ImageNet初始化历史、完整语义neck、scalar目标、256输入和60轮训练预算，不能将收益单因素归因于neck或骨干；旧几何入口768与当前256不满足网络输入identity。
- 旧GeoPRR同人口／ROI谱系及target记录保留，完整旧权重链和执行像素尚未重新认证。旧主Raw为12818候选图／60轮、另3182用于router，独立EfficientNet全16000训练baseline不与旧锚点混同。
- 两外域已查看并参与开发，zero-shot指0目标训练／适配／选择，不能称新盲测、统计显著性或普遍泛化。RF100标量由离线四点派生、zero仅6张。
- 指标中GT辅助逐图最小误差oracle只是描述互补，不是已实现router、融合成绩或未来性能保证。source训练loss按保存摘要检查有限，没有用验证预测重建训练loss。
- 单seed，不扩officialtest或外域epoch搜索。原模型、全部失败／负结果、14494／1506／4000及冻结普通／严重名单保留。公开范围仅[根allowlist](public_allowlist.json)中的标量、逐图、组和metadata，无raw run／source_data_contract、bbox、完整ID人口数组、RGB、权重、H／C／XYZ或特征。
