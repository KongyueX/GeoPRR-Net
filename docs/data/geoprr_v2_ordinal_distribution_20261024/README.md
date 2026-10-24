# V25 有序读数分布源域实验

[独立协议](../../GEOPRR_V2_ORDINAL_DISTRIBUTION_ZERO_SHOT_PROTOCOL_CN.md)与[实际配置](config.json)：fresh完整cachedImageNet MobileNetV4的semantic1280→单个256节点head，唯一稳定左右logcumsumexp CDF Bernoulli log-score，全质量归一期望部署。全部CNN/head源训练、BN仅源更新；无旧task、adapter、query、几何分支或额外loss。

7项新CPU toy检查与真实模型零forward构造通过，实际8762448参数／233参数tensor／465state，head327936参数，全部参数trainable。CPU未加载cached权重或RGB，无GPU／optimizer；实际源AMP见证3CNN／8图／1back／0optimizer已通过，完整cached462 state保留、233参数无optimizer不变、77BN源+1／eval额外0、FP32 logp／CDF／期望及全部必要梯度已核实；正式fresh源训练已启动。见[runtime](runtime_preflight.json)与[启动记录](launch.json)。无外域性能或泛化结论。

完整源停止后只best各一次冻结Industrial／RF100，两域不训练或调参。只显式allowlist规范化标量／元数据公开，RGB、weights、geometry、features、rawrun、fullcontracts和privatepaths私有。
