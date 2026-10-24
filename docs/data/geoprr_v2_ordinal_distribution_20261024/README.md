# V25 有序读数分布：完整结果

[完整报告](completed/report_cn.md)、[完整规范化指标](completed/metrics.json)与[比较表](completed/comparison.csv)已归档。60完整源epoch／source-best37，仅源R选择后两外域各冻结评估一次。Industrial mean21.972040／tail60.546531%FS，较V24改善但明显弱于v16及旧GeoPRR；RF100 mean9.626736／tail42.779526%FS，较v16小幅改善但低阈值准确率下降、max增大，mean／tail仍弱于旧GeoPRR及V24。源拟合改善未达到两个外域整体超越目标，不升级默认。

唯一独立CPU数字／执行核对68292PASS／0fail。完整cached462 encoder state仅在初始化逐项保留；8762448参数／233参数tensor／465 state的全部CNN与有序256节点head共同源训练，BN仅源更新、eval冻结。唯一稳定CDF log-score与部署全质量期望见[独立协议](../../GEOPRR_V2_ORDINAL_DISTRIBUTION_ZERO_SHOT_PROTOCOL_CN.md)、[配置](config.json)、[CPU](cpu_checks.json)、[runtime](runtime_preflight.json)、[启动记录](launch.json)及[显式allowlist](public_allowlist.json)。source_training_cdf_log_score不与旧SmoothL1数值排序。

全部源best／last原1506＋roll512及Industrial1395／RF100151标量、group、history、counts、AMPskip、time与peak保留。所有旧模型／负结果和fixedcohorts保留；RGB、weights、geometry、features、rawrun、fullcontracts、完整源数组和privatepaths不公开。
