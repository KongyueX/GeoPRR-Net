# 固定头warmstart源控制

保持完整MNV4/GAP1280/Linear1、同一source packet与430次全CNN反传，只增加一次冻结encoder的source特征提取和固定100次point校准，再继承同头进行原FT5。只检验source学习流程，不是正式V32、LP-FT论文复现或泛化成功。

[冻结协议](../../GEOPRR_V2_HEAD_WARMSTART_SOURCE_CONTROL_CN.md)明确额外LP成本、不同CNN/point计数、原source-ID学习门槛与条件源OOD；真实Industrial/RF全部调用0。10项NEWCPU检查已通过（1.098892秒，不含imports），真实MNV4构造／forward／cachedweights／RGB／GPU均0，CPU过程CUDA不可见；仅复用已冻结source packet。真实LP/FT零更新pref及正式控制尚未启动；old source packet不重做、旧诊断不重跑，完整source arrays/cache特征/weights/RGB/privatepaths不公开。

启动会先检查冻结snapshot所需metadata，默认拒绝任何已有输出。仅root显式的zero-stage bootstrap恢复可接受同一合同下failed且steps空、无reader/pref、旧parent实际退出且GPU空的情况；原失败记录和部分snapshot完整保存在同output，任何已开始模型阶段均拒绝恢复。该恢复不更改科学配置、预算或训练状态。新增3项针对检查已通过，原10项phase检查未重跑。
