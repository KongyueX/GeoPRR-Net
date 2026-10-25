# V30：因子化二阶语义标量读出

协议见[二阶语义zero-shot协议](../../GEOPRR_V2_BILINEAR_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。保留完整fresh缓存ImageNet MobileNetV4、GAP1、semantic1280及单RGB路径，用64项因子乘积生成唯一scalar logit；encoder FP16、新head FP32，全部参数源训练，BN仅source更新／eval冻结。

本轮不加数据、style算子、几何分支或旧point融合，沿用RGB512、固定源人口／photo／SmoothL1／四项mean-tail源R／min60max160。10项新增CPU检查已通过，含两次真实model零前向／零cachedweights构造及465 strict reload；实际8598481参数／234个trainable参数tensor／465state／462encoderstate，point163969参数／3tensor。真实源AMP见证3CNN／8图／1combinedback／0optimizer已通过，实际semantic1280 FP16／projection128、factor64与logit FP32、两factor与encoderstem／neck梯度有限非零、465strictreload通过。正式源训练已启动，尚无外域结果。原run／fullcontracts／RGB／weights／features／geometry／privatepaths不公开。
