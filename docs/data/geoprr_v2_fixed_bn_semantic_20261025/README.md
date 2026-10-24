# V27：固定ImageNet BN统计的完整语义适配

协议见[固定BN完整语义zero-shot协议](../../GEOPRR_V2_FIXED_BN_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。

V27保留native ROI RGB512、完整fresh缓存ImageNet MobileNetV4 semantic1280、新scalar point及全部参数源训练；固定77个BN的原始运行统计，BN affine继续训练。源人口、SmoothL1、四项mean／tail源R与min60／max160停止规则保持。

本目录只保存显式规范化配置、必要检查、启动与完成工件。9项新增CPU检查及真实模型零前向构造已完成；实际77BN／231运行buffer／154可训练affine tensor／8435793参数／464模型state与声明一致。真实源AMP运行见证、启动和外域结果待记录。原BN参考值、完整源合同／数组、RGB、weights、几何与私有路径不公开。
