# V26 RGB512完整语义源域实验

**源109轮／best89及两域各一次冻结外评已完成；唯一保存数字／执行核对73307项PASS、0失败，不升级默认。** [完整报告](completed/report_cn.md)、[比较表](completed/comparison.csv)、[指标](completed/metrics.json)及[核对记录](completed/numeric_execution_audit.json)保留正负结果。Industrial mean22.861047／tail64.456899%FS，弱于v16、旧GeoPRR与V25；较V24改善。RF100 mean9.059529%FS与Acc@5=71.523179%较v16、V25改善，>5总数43对v16的58（救回30／新增15），但tail51.670712／max70.405841%FS升高。Industrial较V25的>5总数降至833对917（救回191／新增107），同时mean／tail升高。

[独立协议](../../GEOPRR_V2_HIGH_RESOLUTION_DIRECT_ZERO_SHOT_PROTOCOL_CN.md)与[实际配置](config.json)记录单nativeROI RGB512→完整fresh cached ImageNet MobileNetV4→semantic1280→唯一new point／FP32 sigmoid。实际8435793参数／233参数tensor／464state，全部CNN／point源训练，BN仅源更新、eval冻结；cached462保留只指初始化。相同nativeROI／标签／光度／roster，权威输入已为512／128，旧256／64只为population provenance；不能称RGB合同等于V16／V25。没有CDF、adapter、query或几何分支。

完整源R与min60／max160规则选择best89；109−89=20结束，无resume，last109只源。原1506及roll512的best／last四份完整标量CSV均保留，roll512只自身报告。两域各一次冻结同一best，目标训练／统计拟合／BN适配／超参或checkpoint选择均0。两域为已查看开发人口，单seed不支持SD、新盲测、普适性或纯512因果。

正式CNN／point各4033674图／63136调用，59623次源backward尝试＝59600更新＋23AMPskip；pref3CNN／8图／1back／0optimizer单列。训练pipeline18503.923992秒、source full19355.291917秒，source peak allocated／reserved8262538752／11347689472bytes；外评、pref和历史成本见[成本](completed/costs.json)。一次CNN不等于相同成本。

[completed显式allowlist](completed/public_allowlist.json)26项，[总allowlist](public_allowlist.json)33项。只公开规范化保存标量／metadata；全分母逐图CSV、paired counts、完整scalar group summaries及必要focus／maxIDs保留。rawrun、fullcontracts、完整源人口／fixed-cohort ID数组、per-image delta maps、完整improved／regressed与threshold／rescue／new ID列表、RGB、weights、geometry、features、bbox、原BN值和privatepaths保持私有。旧GeoPRR历史pixels／完整权重链未重新认证；本次核对未加载checkpoint内容。四阶段complete／exit0，收集时父进程退出／GPU空闲。所有旧模型和失败保留，无自动promote。
