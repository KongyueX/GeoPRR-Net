# v19固定C4轨道语义池化

**15轮与两套冻结外评完整结束，source-best6源NMAE为25.208096%FS，Industrial为39.098746%FS，RF100为24.328313%FS。** 99.0027%的有限源预测落在0.5±0.01，源均值未优于固定0.5常数25.165799%FS。这一配方没有学好源读数，也没有改善zero-shot；完整负结果保留，不作为默认模型，不延长训练。

每张256 ROI固定生成0／90／180／270度四个张量旋转，共享完整ImageNet MobileNetV4，在1280维semantic上取FP32均值，再执行point和sigmoid。训练、验证、外域使用同一函数，参数8435793／464 state，新增参数0。源18590／验证1506保持，只在完成全部15轮后各评估一次冻结source-best6；Industrial1395与RF100151没有用于训练或适配。

正式17430次反传尝试／17423次有效更新／7次AMP跳过，1211944 CNN视图／18953次调用，目标反传与更新均0。训练CNN视图数与v16相同，但15轮独立光度抽样、checkpoint机会、相关方向BN batch和四视图部署成本不同。

源1506、Industrial1395、RF100151分别保留2／19／1个无效scalar，每个原图按error=1计分，输入失败均0。一次源16图批次的AMP／FP32诊断确认一个AMP激活非有限样本在FP32下恢复有限但仍预测错误，其余15图仍接近0.5；诊断没有更改正式预测或精度策略。保存工件的一次数值／执行审计27706项通过、0失败。

- [完整报告](completed/report_cn.md)
- [完整指标](completed/metrics.json)
- [模型比较](completed/comparison.csv)
- [源精度诊断](completed/source_precision_probe.json)
- [实际成本](completed/costs.json)
- [审计](completed/numeric_execution_audit.json)
- [冻结协议](../../GEOPRR_V2_C4_ORBIT_ZERO_SHOT_PROTOCOL_CN.md)
- [启动记录](launch.json)
- [实际源运行检查](runtime_preflight.json)
- [新CPU检查](cpu_checks.json)

## Limitations

本次只约束固定C4 semantic均值配方；整数90度整图roll不覆盖所有透视、刻度方向或连续旋转。源训练拟合不足与至少一个已检查的AMP数值问题并存，不能从一个源批次推断全部无效预测的原因。两外域已查看、单seed，不称新盲测或显著性。完整旧模型和负结果保留，RGB／权重／原始几何与合同不公开。
