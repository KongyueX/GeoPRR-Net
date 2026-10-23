# MobileNetV4冻结特征直接读出：完整负结果

2026-10-23。**v15新增直接读出在源验证就没有学好：source-best19为17.178667%FS，同次原生几何读出为0.063613%FS；两个外域和固定0.5输出均值也明显回退。** 这不是“源已学好、仅zero-shot翻车”的结果。它只检验冻结feedback48的960维GAP特征接线性sigmoid小头，不否定端到端直接进度训练或旧Raw路线。

| source-best19对应readout | 源1506 NMAE %FS | Industrial1395 NMAE %FS | RF100151 NMAE %FS |
|---|---:|---:|---:|
| 原生geometry，任务权重始终为feedback48 | 0.063613 | 17.862102 | 10.938938 |
| direct，新增961参数sigmoid线性头 | 17.178667 | 39.015321 | 38.410688 |
| 预先固定的0.5输出算术均值 | 8.591947 | 26.078151 | 23.730463 |

完整20轮CPU head预算结束，源验证最优为第19轮，末轮20另存；只对source-best19作两套外评，没有按Industrial／RF100拟合统计量、选择epoch或参数。这里zero-shot指0目标训练／适配，两外域已被查看，不称新盲测。两个输出的算术均值区别于v14权重插值；三个readout共享同一次CNN前向，不是三次CNN计算。

[完整报告](completed/report_cn.md)、[全部指标与paired ID](completed/metrics.json)、[LF比较表](completed/comparison.csv)、[142946项数值／执行核对](completed/numeric_execution_audit.json)和[16文件完整结果清单](completed/public_allowlist.json)均已归档。source-best19与last20 combined逐图各1506行，两个外域combined逐图1395／151行；每行分别保留geometry、direct、fixed half的标量、有效性、误差及失败信息。普通1495／1497、严重6、3172／2810／3663／9323、全部group／component、zero／nonzero／nearzero、Acc、tail／max和救回／新增均按原人口保存。

## 输入、统计量与执行

原生模型完整496 state／7461060参数、所有参数与BN buffers均冻结eval；在最后一个实际CNN特征层取GAP960维，新head为sigmoid(linear(标准化特征))，零weight／bias初始化。固定mean／std只由18590×960源训练特征计算，不使用1506验证或任何外域统计。独立CPU proof确认保存统计量与训练精确相同，并以F.linear＋sigmoid复算source-best19验证预测，最大差严格0；该proof不重载native CNN，不执行CNN／反传／更新。见[train-only统计量证据](completed/source_statistics_proof.json)。

正式CNN实际21642张图／774次调用，CNN backward／optimizer均0；CPU新head20轮呈现371800条源训练特征，实际1460次反传／1460次optimizer更新，不能将整个阶段称为0更新。源cache、head fit、Industrial、RF100队列wall分别153.621350／7.320929／18.650663／13.803992秒，完整队列193.554400秒。必要见证1次CNN／2张真实源图、1次CPU head反传、0optimizer单独保存于[运行见证](runtime_preflight.json)，[4项CPU实现检查](cpu_checks.json)也保留原文件。归档没有重复执行原模型、旧GT、教师或评分核对。

RGB和head仅使用既有端点替换源18590 unique与原验证1506；cache无光度增强、原768／192、ImageNet／AMP，训练cache batch32／val16／workers4。head固定CPU20轮、batch256、AdamW、normalized L1，预算不冒充旧donor／recipient全量微调。外域是原生ROI RGB，同一配置、全分母error1，无目标几何输入、BN／统计量适配、ROI／阈值／温度／TTA调整。

## 与旧GeoPRR的区别

旧clean历史同人口最终均值为Industrial11.643618／RF1006.607079%FS，但Raw base本身已为11.642999／6.569962%FS；两套clean人口全部relation_invalid，relational等于base，最终融合也未优于base。该优势不能直接记给SARN、polar或router。旧Raw表示曾以直接进度目标训练，把输出层接到当前几何专门化冻结特征并不复现其训练路线。

旧主模型Raw实际用12818候选训练图、60轮、12058有效更新，源池16000中另3182给router；独立EfficientNet全16000 baseline属于不同合同，不与旧GeoPRR锚点混同。旧权重链当前不可重放、历史执行像素完全相同未重新认证，保留这一边界。协议及证据链接见[直接读出协议](../../GEOPRR_V2_DIRECT_READOUT_ZERO_SHOT_PROTOCOL_CN.md)。

## 下一独立方案

用户要求继续提高0目标监督的泛化，不限于旧GeoPRR。v16已冻结一个独立方案：使用同一已有ImageNet缓存的完整MobileNetV4分类语义通路，保留features_only丢弃的conv_head960→1280及norm_head六个state／1231360参数；离线CPU构造确认实际embedding1280，去分类器encoder8434512参数／462state，新增1281参数point head后8435793参数／464state。新方案为256 RGB、源端到端60轮、SmoothL1(beta0.05)，不是延长v15的缓存线性头，也不使用Industrial或RF100训练／适配。

当前是已冻结计划，新的实际256源输入运行见证与正式训练尚未开始，没有新性能结论。详见[完整语义源域训练协议](../../GEOPRR_V2_FULL_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。

## Limitations

本轮只否定既定冻结GAP线性头配置的收益，不能证明直接RGB路线本身无效。外域已查看、单seed；0目标监督不等于未触碰盲测，不称显著性、普适泛化或新算法认证。旧GeoPRR训练史、模型表示与预算不同，历史像素／权重链没有重新认证。源中间epoch只保留history摘要，best19／last20逐图完整保存；RF100标量由离线四点派生、zero只有6张。原模型、全部旧失败／负结果、原14494／1506／4000及冻结名单保留。公开仅[根清单](public_allowlist.json)中的规范化标量、逐图和CPU metadata；无RGB、特征／统计向量、head／native权重、raw H／data_contract或private trace。
