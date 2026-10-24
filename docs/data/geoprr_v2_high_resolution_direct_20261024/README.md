# V26 RGB512完整语义源域实验

[独立协议](../../GEOPRR_V2_HIGH_RESOLUTION_DIRECT_ZERO_SHOT_PROTOCOL_CN.md)与[实际配置](config.json)记录单nativeROI RGB512→完整freshcachedImageNet MobileNetV4→semantic1280→唯一newpoint/sigmoid。完全复用源人口与原nativeROI／标签／光度，实际RGB512/128合同改变，输入元素为256的四倍，不称同成本。全部CNN/point源训练，BN仅源更新eval冻结；无CDF／adapter／query／几何分支。

6项新CPU接口／数据合同toy检查及真实模型零forward构造通过，实际8435793参数／233参数tensor／464state，全部trainable；CPU未加载cached权重或真实RGB，无GPU／optimizer。真实512源AMP见证及正式训练尚未执行，不预报性能。

两个已查看外域不训练或调参，只在完整source停止后各冻结best一次。只explicitallowlist规范化标量／元数据公开，原RGB、weights、geometry、features、rawrun、fullcontracts和privatepaths私有。
