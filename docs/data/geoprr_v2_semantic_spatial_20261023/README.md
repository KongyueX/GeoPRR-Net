# v17完整语义与空间残差读出：完整结果

2026-10-23。**已完整跑完8轮，best8=last8，正常退出；空间残差仅让源均值改善1.2892%，Industrial与RF100均小幅回退，没有zero-shot增量。** 不提升v17、不延长8轮，全部父模型、新权重与负结果保留。

| 模型 | 源1506 NMAE %FS | Industrial1395 NMAE %FS | RF100151 NMAE %FS | 源max %FS |
|---|---:|---:|---:|---:|
| v16 parent best59 | 0.526217 | 11.686054 | 10.164776 | 65.953931 |
| v17 best8=last8 | 0.519433 | 11.690235 | 10.182052 | 65.944996 |
| 旧GeoPRR clean历史 | — | 11.643618 | 6.607079 | — |
| v12 local端点best1 | 0.083392 | 20.539125 | 7.659157 | 18.677585 |

相对v16，源下降0.006784个百分点，Industrial上升0.004181个百分点、RF100上升0.017276个百分点；后两者相对上升0.03578%／0.16996%。源>1／2／5%FS总数仍127／40／13，大错没有解决：8195仍65.944996%FS，3172为15.921795%FS、3663为8.803230%FS。RF100的046与006仍分别错54.482259／46.066302%FS。

[完整报告](completed/report_cn.md)、[全量指标与paired ID](completed/metrics.json)、[比较表](completed/comparison.csv)、[8轮history](completed/history.jsonl)、[53401项数值执行核对](completed/numeric_execution_audit.json)和[18文件清单](completed/public_allowlist.json)已归档。普通1495／1497、严重6、3172／2810／3663／9323／8195、zero／nonzero／nearzero、Acc、tail／max、所有组和救回／新增名单均保留。

## 实际结构与执行

父v16 best59的464个state、8435793参数及77个BN counter完整冻结，8／8轮保存记录均不变。新增空间分支129057参数／10state，完整8564850参数／474state；同一次CNN前向读取stride8的80×32×32局部特征，形成3个通用空间token和logit残差。模型输出确实改变，源／Industrial／RF100平均绝对预测变化0.070459／0.076957／0.072720%FS，任何一张均未改变超过1%FS，是小幅修正而非大错修复。

固定8完整源epoch、18590训练／1506验证、相同256 RGB／LINEAR／ImageNet，scalar SmoothL1 beta0.05，只训练新分支。epoch0复用已审计v16源基准，源选择最终为8，完成后两域各真实评估一次，0目标训练／统计拟合／BN适配／checkpoint选择。

正式162314张CNN图像／2545次调用：训练148720／2328、源验证12048／192、外评1546／25。branch实际2328次反传／更新、AMP skip0，CNN反传和父参数更新0，外域反传／更新0；无resume、新增渲染0。训练完整process402.939907秒，Industrial／RF100为11.327447／10.508867秒。4项CPU检查、4张unique源RGB／8图前向的0更新运行见证单列并保留原文件。

best8和last8的解压CSV字节精确相同，gzip容器字节不同；公开一份[共享源CSV](completed/source_best8_last8_predictions.csv.gz)并保留[别名记录](completed/source_aliases.json)。这只是源保存表去重，两个外域均为新模型实际前向，没有复用父模型的外评分数。完成核对53401项PASS、0失败、最大数值差1.42e-14，没有追加模型或GT核验。

下一结构尚未选择或冻结。本轮是有界的空间分支负结果，不默认继续叠加结构。评分前冻结配方与实际输入记录见[协议](../../GEOPRR_V2_SEMANTIC_SPATIAL_ZERO_SHOT_PROTOCOL_CN.md)、[运行见证](runtime_preflight.json)、[CPU检查](cpu_checks.json)和[启动记录](launch.json)。

## Limitations

- 该结果针对冻结父模型的这一源训练空间残差配方，不是端到端attention，也不说明attention普遍无效；3个token未认证为针尖或刻度角色。
- 同256源输入合同与父模型精确对齐，冻结state／BN根据保存runtime证据核对，完成审计没有重新加载checkpoint或使用GPU。
- 两外域已查看，0目标训练／选择不等于新盲测；单seed和微小差异不作统计显著性声明。RF100 scalar由四点离线派生、zero仅6张。
- 旧GeoPRR历史同人口／ROI谱系保留，但执行像素与完整权重链尚未重认证；旧clean优势在Raw base已存在，不归给SARN／polar／router。
- 原模型、全部失败与冻结人口保持。公开仅[根allowlist](public_allowlist.json)中的标量、逐图、组和metadata，无raw run／data_contract、private paths／bbox、RGB、权重、H／C／XYZ或特征。
