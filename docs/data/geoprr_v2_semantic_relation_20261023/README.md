# v20完整语义条件关系读出

**60轮和两套冻结外评完整结束，source-best16源NMAE为0.283315%FS，Industrial为29.257524%FS，RF100为16.224095%FS。** 源均值比v16／v18低46.1601%／36.5618%，两个外域均更差，没有zero-shot增量。本轮不作为默认模型、不延长60轮，完整资产与负结果保留。

完整ImageNet MobileNetV4的semantic1280条件预测校准，由共享多尺度FPN与local参考角色联合J执行原CircularProgressPushforward全质量读数；训练／验证／外域均部署该通路。实际8762052参数／502 state，没有直接point、融合router或CDF。forward只接收RGB，H／span／解析U／M和虚拟角色标签只在源loss中使用。

只用18590源训练／原1506源验证，fresh缓存ImageNet、256 LINEAR／ImageNet、原光度、seed20262020和60完整epoch；最早源NMAE最小值出现在第16轮，全部60结束后才各做一次冻结外评。Industrial1395与RF100151没有参与训练、统计拟合、BN适配或选择。输入与scalar全部有效，coverage1，没有删除失败样本。

实际17460次反传尝试／17453次更新／7次AMP跳过，无resume；正式1207306 CNN图／18925次调用。关系头和读出分别执行17460次训练、1440次验证、25次目标调用，目标反传／更新0。一次保存工件数值／执行审计54062项通过、0失败；源best／last各1506行和两外域完整逐图均已归档。

- [完整报告](completed/report_cn.md)
- [完整指标](completed/metrics.json)
- [模型比较](completed/comparison.csv)
- [实际成本](completed/costs.json)
- [审计](completed/numeric_execution_audit.json)
- [冻结协议](../../GEOPRR_V2_SEMANTIC_RELATION_ZERO_SHOT_PROTOCOL_CN.md)
- [源映射兼容性摘要](source_role_mapping_feasibility.json)
- [新CPU检查与实际模型构造](cpu_checks.json)
- [实际源RGB运行见证](runtime_preflight.json)
- [正式队列启动记录](launch.json)
- [公开allowlist](public_allowlist.json)

## Limitations

源均值改善伴随少量大错，best16最大53.666402%FS（sync_9），last60最大88.976995%FS（sync_4025）；不能据源更准宣称泛化更好。理想源标签映射兼容性不是可学习性能或新理论，outside_arc_mass不能硬门控准确率。已查看的两外域、单seed不构成新盲测或显著性证据；旧历史模型输入与训练史不同。原14494／1506／4000、普通／严重名单、全部旧失败保留，RGB／权重／原始几何与合同不公开。
