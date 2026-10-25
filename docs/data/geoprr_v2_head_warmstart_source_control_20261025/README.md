# 固定头warmstart源控制

保持完整MNV4/GAP1280/Linear1、同一source packet与430次全CNN反传，只增加一次冻结encoder的source特征提取和固定100次point校准，再继承同头进行原FT5。只检验source学习流程，不是正式V32、LP-FT论文复现或泛化成功。

[冻结协议](../../GEOPRR_V2_HEAD_WARMSTART_SOURCE_CONTROL_CN.md)明确额外LP成本、不同CNN/point计数、原source-ID学习门槛与条件源OOD；真实Industrial/RF全部调用0。10项NEWCPU检查已通过（1.098892秒，不含imports），真实MNV4构造／forward／cachedweights／RGB／GPU均0，CPU过程CUDA不可见；仅复用已冻结source packet。真实LP/FT零更新pref及正式控制尚未启动；old source packet不重做、旧诊断不重跑，完整source arrays/cache特征/weights/RGB/privatepaths不公开。
