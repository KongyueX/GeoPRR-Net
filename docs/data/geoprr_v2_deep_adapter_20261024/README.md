# V24 深层adapter：完整结果

[完整报告](completed/report_cn.md)、[完整规范化指标](completed/metrics.json)与[比较表](completed/comparison.csv)已归档。98完整源epoch／source-best78，仅源R选择后两外域各冻结评估一次。Industrial mean23.980845／tail65.664809%FS；RF100 mean7.726820／tail25.325275%FS。RF相对v16 mean／尾部改善，但Industrial退化、RF低阈值准确率更差，未达到整体超越目标，不升级默认。

唯一CPU数字／执行核对72401PASS／0fail。完整原462state与77BN在全部源epoch及eval保持，476281个新adapter+point参数参与训练。详见[冻结协议](../../GEOPRR_V2_DEEP_ADAPTER_ZERO_SHOT_PROTOCOL_CN.md)、[配置](config.json)、[CPU](cpu_checks.json)、[runtime](runtime_preflight.json)、[启动](launch.json)及[显式allowlist](public_allowlist.json)。

所有旧模型／负结果和fixedcohorts保留；原RGB、权重、geometry、features、rawrun、fullcontracts、完整源数组和privatepaths不公开。
