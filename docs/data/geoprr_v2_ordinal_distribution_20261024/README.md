# V25 有序读数分布源域实验

[独立协议](../../GEOPRR_V2_ORDINAL_DISTRIBUTION_ZERO_SHOT_PROTOCOL_CN.md)与[实际配置](config.json)：fresh完整cachedImageNet MobileNetV4的semantic1280→单个256节点head，唯一稳定左右logcumsumexp CDF Bernoulli log-score，全质量归一期望部署。全部CNN/head源训练、BN仅源更新；无旧task、adapter、query、几何分支或额外loss。

7项新CPU toy检查与真实模型零forward构造通过，实际8762448参数／233参数tensor／465state，head327936参数，全部参数trainable。CPU未加载cached权重或RGB，无GPU／optimizer；实际源AMP运行见证及正式训练尚未执行，不预报泛化。

完整源停止后只best各一次冻结Industrial／RF100，两域不训练或调参。只显式allowlist规范化标量／元数据公开，RGB、weights、geometry、features、rawrun、fullcontracts和privatepaths私有。
