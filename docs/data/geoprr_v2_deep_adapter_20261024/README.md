# V24 深层adapter源域实验

[冻结协议](../../GEOPRR_V2_DEEP_ADAPTER_ZERO_SHOT_PROTOCOL_CN.md)与[实际配置](config.json)记录新独立表示实验。原完整ImageNet权重与BN固定，仅训练两个内部残差adapter及新scalar point。数据／loss／源R与V23保持；两个已查看外域只在完整source停止后各冻结评估一次。

6项新CPU toy检查与真实model零forward构造已通过，实际8910793参数／476281可训练参数／241个参数tensor（10个可训练）／472个state。真实源RGB执行见证与正式训练尚未执行；不预报效果。

仅本目录显式allowlist的规范化标量／元数据公开，原RGB、权重、geometry、features、rawrun、fullcontracts和privatepaths保持私有。
