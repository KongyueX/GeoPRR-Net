# V28：2×2空间语义直接读数

协议见[空间语义zero-shot协议](../../GEOPRR_V2_SPATIAL_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。保留完整cached ImageNet MobileNetV4，将末端pool由1×1改为固定2×2，原完整1280 neck作用于四cell，flatten5120接唯一新scalar point；全部CNN／point源训练，BN仅source更新。

本轮不加数据或几何分支，沿用RGB512、固定源人口／photo／SmoothL1／四项mean-tail源R／min60max160。5项新增CPU检查和真实model零前向构造已通过，确认8439633参数／233参数tensor／464state、1280每cell／5120读出、pool0参数／0state。真实源AMP见证3CNN／8图／1combinedback／0optimizer已通过，实际四cell neck与5120读出、cached462初始化、必要梯度及464strict reload通过；正式源训练已启动，尚无外域结果。原run／contracts／RGB／weights／geometry／feature／privatepaths不公开。
