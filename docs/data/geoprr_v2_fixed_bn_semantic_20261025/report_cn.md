# V27固定BN完整语义适配：源训练失败记录

**V27未完成一个源epoch。正式训练在epoch1 batch36触发FloatingPointError: nonfinite scalar loss，源阶段exit1，Industrial／RF100均未启动。** 失败时间2026-10-24T21:05:28.692328+00:00。没有source best、latest、history或completed记录，因此没有源验证R、冻结外域分数或可用的严格resume checkpoint。本轮作为真实源训练失败保留，不升级默认。

| 阶段 | 保存结果 | 结论 |
|---|---|---|
| 真实源AMP零更新见证 | PASS；3CNN／8图／1combinedback／0optimizer | 该固定输入上的执行与梯度见证通过 |
| 正式源首批 | epoch1 batch1；64图；loss0.21415004134178162；LR0.0003；gradient norm finite | 正式源训练已实际开始 |
| 正式源失败 | epoch1 batch36；nonfinite scalar loss；exit1 | 完成源epoch为0 |
| Industrial／RF100 | 均未启动 | 没有外域结果或比较 |
| 保存时执行状态 | parent与trainer退出；GPU compute processes为0 | V27已终止 |

真实源见证记录实际RGB512／output128、完整semantic1280及单scalar point；77个BN共231个running_mean／running_var／counter buffer在源train与validation前后逐项保持原缓存ImageNet值，154个BN affine gradient tensor有限非零，CNN／stem／完整neck／point梯度有限非零。参数233 tensor在零optimizer下保持，464模型state strict reload通过。pref与正式训练fresh初始化分开。以上是[保存见证](runtime_preflight.json)，不能替代持续优化后的源或外域结果。

实际模型8435793参数／233参数tensor／464state。完整cached encoder462 state只在fresh初始化逐项保留；卷积、BN affine与point均允许源训练，固定的是原BN运行统计。输入、源人口／ROI／scalar标签／原光度、SmoothL1、源R和停止合同见[配置](config.json)与[协议](../../GEOPRR_V2_FIXED_BN_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。

| 执行计数范围 | 数量 | 证据边界 |
|---|---|---|
| pref CNN／point | 各3调用／8图 | 保存counter |
| pref combined backward／optimizer | 1／0 | 保存counter |
| 正式source optimizer updates | null | 没有完整epoch／checkpoint保存总计；未知，不是0 |
| 正式source AMP skips | null | 没有完整epoch／checkpoint保存总计；未知，不是0 |
| 正式source backward／forward部分路径 | 失败前35次完整backward尝试，第36批forward触发loss失败；各批64图 | code-implied partial信息，不是保存counter观测总量或正式完整预算 |
| target训练／适配 | 0 | I/R均未启动 |

源训练自2026-10-24T21:04:58.140152+00:00启动，2026-10-24T21:05:28.692328+00:00以exit1结束。保存执行检查时间2026-10-24T21:07:49.360497+00:00，parent／trainer命令均不存在，GPU进程为空。pref process／full为20.476757999509573／21.42031529545784秒，peak allocated／reserved630159872／725614592bytes；这些只属于pref，不能当作失败源训练的完整成本。

原配方、配置、执行快照、raw run与日志及全部历史资产保留。没有同配方重启，没有改变LR／精度或放松BN统计保护；目标训练、统计拟合、BN适配及目标checkpoint／超参选择均未发生。[失败记录](failure_record.json)与[9项显式allowlist](public_allowlist.json)保存必要公开metadata；rawrun、fullcontracts、完整源数组、raw logs／traceback、RGB、weights、geometry、features、bbox、原BN值和privatepaths留私有。

## Limitations

观察到的失败是nonfinite scalar loss，根因尚未确定，不能写成已确认FP16 overflow或BN策略失效。一次零更新见证和有限首批loss不能证明持续训练稳定，也不能从不完整源训练推断泛化效果。单seed、源域partial记录不支持完成阶段指标、模型优劣、显著性或因果结论。没有完整四阶段结果，不生成完整阶段数字审计或外域比较。
