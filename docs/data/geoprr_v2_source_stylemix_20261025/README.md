# V29：单早期层源特征风格混合

协议见[源特征风格混合zero-shot协议](../../GEOPRR_V2_SOURCE_STYLEMIX_ZERO_SHOT_PROTOCOL_CN.md)。完整fresh缓存ImageNet MobileNetV4、GAP1、semantic1280与新scalar point，在一个早期层训练时混合源mini-batch逐图channel统计；eval精确identity，不计算或拟合目标统计。

固定p1／α.1／ε1e-6／FP32 moments／逐图空间方差（unbiased=False），源人口、RGB512、原photo、SmoothL1、源R及min60max160保持。7项新增CPU检查及真实model零前向构造已通过，确认8435793参数／233个trainable参数tensor／464state／462encoderstate／77BN，style算子0参数／0state。真实源AMP见证3CNN／8图／1combinedback／0optimizer已通过；实际source style active1次4图、eval inactive2次4图且精确identity，FP32 moments／必要梯度／464strictreload通过。正式源训练已启动，尚无外域结果。rawrun／fullcontracts／RGB／weights／feature／moments值／geometry／privatepaths不公开。
