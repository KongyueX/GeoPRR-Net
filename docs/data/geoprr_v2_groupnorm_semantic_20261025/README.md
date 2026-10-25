# V31：单图分组归一化语义读出

协议见[GroupNorm语义zero-shot协议](../../GEOPRR_V2_GROUPNORM_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。完整缓存MobileNetV4内77个BN改为固定8组GN，复制原affine并保留drop→act，原学习参数保持、BN运行buffer移除；GAP1280和新单scalar point，全部参数源训练。

GN训练／eval均按单张图内部计算，不在目标人口上拟合统计或适配参数。固定原eps1e-5、FP32 norm数学／原feature dtype输出。源人口、RGB512、photo、SmoothL1、R与min60max160保持。12项新增CPU检查1.658446427s已通过，含两次真实NEW GN model零前向／零cachedweights构造和233 strict reload；实际8435793参数／233个trainable参数tensor／233state／GN77／BN0／buffer0，原231学习参数保持、231BN运行buffer移除。真实源AMP见证、启动和外域结果待记录。原run、contracts、RGB、weights、features、统计实际值及privatepaths不公开。
