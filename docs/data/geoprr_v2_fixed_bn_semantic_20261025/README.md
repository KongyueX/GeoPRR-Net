# V27：固定ImageNet BN统计的完整语义适配

**V27正式源训练在epoch1 batch36因nonfinite scalar loss失败，完成源epoch为0；Industrial／RF100均未启动，没有外域结果。** 源阶段exit1，父进程与trainer已退出，保存检查时GPU空闲。没有history、latest、best或completed checkpoint，不能严格resume。见[失败报告](report_cn.md)、[失败记录](failure_record.json)与[启动／终止记录](launch.json)。

[协议](../../GEOPRR_V2_FIXED_BN_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)冻结的配方为native ROI RGB512、完整fresh cached ImageNet MobileNetV4 semantic1280、新scalar point与全部参数源训练；77个BN的原231运行buffer固定，154个BN affine tensor继续训练。实际8435793参数／233参数tensor／464state；源人口、SmoothL1、四项mean／tail源R与min60／max160停止规则保持。

[真实源AMP零更新见证](runtime_preflight.json)通过：3CNN／8图／1combinedback／0optimizer，231 BN buffer逐项保持、154 BN affine gradient tensors有限非零，464 strict reload。正式首批64图loss0.21415004134178162、LR3e-4、gradient norm finite；随后真实源训练失败。零更新见证不提供完整源训练或外域性能结论。

未完成epoch，实际optimizer updates和AMP skips没有保存总计，均记null，不记0。代码路径只推出失败前35次完整backward尝试及第36批64图forward失败；这是partial code-implied信息，不是保存counter观测总量或正式完整预算。全部既有资产保留，不用同配方重启，不更改LR／precision或放松统计保护，不自动升级模型。

[显式allowlist](public_allowlist.json)共9项，只公开必要规范化配置与执行／失败metadata。rawrun、fullcontracts、完整源数组、raw logs／traceback、RGB、weights、geometry、features、bbox、原BN参考值及privatepaths保持私有。
