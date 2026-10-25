# V28：2×2空间语义直接读数

**源68轮／best48及两域各一次冻结外评已完成；唯一保存数字／执行核对69045项PASS、0失败，不升级默认。** [完整报告](completed/report_cn.md)、[比较表](completed/comparison.csv)、[指标](completed/metrics.json)与[核对记录](completed/numeric_execution_audit.json)保留正负结果。Industrial mean12.758458／tail44.068174／max71.893692%FS较V26均改善；mean仍弱于v16与旧GeoPRR，tail低于v16而略高于旧GeoPRR，max低于两者。RF100 mean10.831387／tail54.135926%FS弱于v16、旧GeoPRR、V24、V25与V26；Acc@2／5较v16改善，>5总数56对58（救回17／新增15），max68.525833%FS较V25与V26降低。

[协议](../../GEOPRR_V2_SPATIAL_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)与[配置](config.json)记录单nativeROI RGB512→完整cached ImageNet MobileNetV4→无参数pool2×2→保留neck1280×2×2→NCHW flatten5120→唯一新scalar point。实际8439633参数／233参数tensor／464state，pool0参数／0state；encoder8434512参数／462state初始化保留，全部CNN／BN affine／point随后源训练、BN只source更新／eval冻结。1280每cell与5120读出分别记录，不是V26的GAP1280或V27固定BN。

权威输入保持V26的RGB512／output128／nativeROI／标签／photo／roster；旧256／64只为population provenance，不称RGB合同等于V16／V25。完整源R与min60／max160选best48，68−48=20结束，无resume，last68只源。原1506＋roll512的best／last四份完整scalar CSV均保留，roll512只自身报告。两域只在源完整结束后各一次冻结同一best，目标训练／统计拟合／BN适配／checkpoint或超参选择均0。V27没有外域结果，保留为失败记录。

正式CNN／point各2517002图／39397调用，37196次源backward尝试＝37182更新＋14AMPskip；pref3CNN／8图／1back／0optimizer单列。训练pipeline11355.169516秒、source full11874.171350秒，source peak allocated／reserved8262600192／11347689472bytes；外评、pref与保存历史成本见[成本](completed/costs.json)。四cell neck／5120point虽单CNN但不是同成本。

[completed allowlist](completed/public_allowlist.json)26项，[总allowlist](public_allowlist.json)33项。全分母scalar CSV、paired counts、完整scalar group summaries与必要focus／maxIDs保留；rawrun、fullcontracts、完整人口／fixed-cohort ID数组、per-image delta maps、完整improved／regressed与threshold／rescue／new名单、RGB、weights、geometry、features、bbox、原BN值及privatepaths私有。单seed与两个已查看开发域不支持SD、新盲测、普适性或纯pool因果；旧GeoPRR历史pixels／完整权重链未重新认证，本次未加载checkpoint内容。四阶段complete／exit0，保存检查时父进程退出／GPU空闲。全部旧模型与正负资产保留，无自动promote。
