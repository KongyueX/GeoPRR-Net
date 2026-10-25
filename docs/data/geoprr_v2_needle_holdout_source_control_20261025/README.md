# 五轮源针型留出控制

该控制保持V26完整MNV4／GAP1280／单scalar读出，只检验在固定小预算内能否学会两种已见合成针型，以及在先满足源ID可学习性后，同一last5对留出第三种既有针型的误差。针型由source metadata字典序预先选择，训练5496既有RGB每轮一次，ID292／条件OOD220，固定5轮。真实Industrial／RF外域不读取、不评估，也不参与任何选择。

源依据、固定人口、constant0.5双门槛和计算预算见[冻结协议](../../GEOPRR_V2_NEEDLE_HOLDOUT_SOURCE_CONTROL_CN.md)。这是问题定位控制，不是正式V32或泛化候选。22项新CPU selector／closure／gate／conditionalOOD检查已通过，真实模型构造／forward／RGB／cachedweights／GPU调用均0；一次source metadata准备完成，5496／292／220精确manifest冻结在服务器，0RGB／模型／GPU。一次source-ID-only零更新pref和正式执行尚未完成。未通过必要条件不得启动GPU，不自动增加轮数、seed或新权重。完整source名单、RGB、weights、features、geometry与privatepaths不公开。
