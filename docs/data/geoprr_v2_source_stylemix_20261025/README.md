# V29：单早期层源特征风格混合

**源60轮／best15及两域各一次冻结外评已完成；唯一保存数字／执行核对68816项PASS、0失败，不升级默认。** [完整报告](completed/report_cn.md)、[比较表](completed/comparison.csv)、[指标](completed/metrics.json)与[核对记录](completed/numeric_execution_audit.json)保留正负结果。Industrial mean22.354576／tail78.790409／max97.788346%FS，mean较V26小幅降低、tail／max升高，三项均弱于V28；相对V25 >5救回186／新增143，但mean／tail升高。RF100 mean11.805282／tail51.893965／max72.905040%FS，mean弱于主要参照；tail较V28降低，同时mean／max升高。RF相对V24 >5救回39／新增16，不能替代均值与尾部代价。

[协议](../../GEOPRR_V2_SOURCE_STYLEMIX_ZERO_SHOT_PROTOCOL_CN.md)与[配置](config.json)记录完整fresh cached ImageNet MobileNetV4、GAP1／semantic1280／新point1280→1；实际8435793参数／233个trainable参数tensor／464state。cached462只在初始化保留，全部CNN／point随后源训练、BN仅source更新／eval冻结；不是V28的2×2／5120读出。唯一encoder.blocks[1]算子训练p1／α.1／ε1e-6，FP32 detached逐图channel μ／σ、variance unbiased=False；eval精确identity，不计算或缓存目标统计。该固定变体参考MixStyle作方法动机，不是原论文默认配置的精确复现或本任务收益证明。

权威输入保持RGB512／output128／原ROI／标签／photo／roster，旧256只为population provenance。完整源R选择best15，min60使第60轮停止，best年龄45；LR下降不重置counter，无resume。last60只源；两域只在完整源停止后各一次冻结同一best，目标训练／统计拟合／BN适配／checkpoint或超参选择均0。两源best／last各原1506＋roll512 CSV完整保留，roll512只自身报告；V27无外域结果，不作为baseline。

正式CNN／point各2221066图／34765调用，32820次源backward尝试＝32806更新＋14AMPskip。style source train active32820次／2098440图，source validation及target合计inactive1945次／122626图；operator forward51.958477784秒，已包含在pipeline且不含backward。pref3CNN／8图／1back／0optimizer及style active1／4、inactive2／4单列。训练pipeline10120.079081秒、source full10590.045265秒，source peak allocated／reserved8262579712／11345592320bytes；全部[实际成本](completed/costs.json)保留，一次CNN不等于同成本。

[completed allowlist](completed/public_allowlist.json)26项，[总allowlist](public_allowlist.json)33项。完整scalar CSV、paired counts、scalar group summaries与必要focus／maxIDs保留；rawrun、fullcontracts、完整人口／fixed-cohort名单、per-image delta maps、完整improved／regressed与threshold／rescue／new名单、RGB、weights、moments值、features、geometry、bbox及privatepaths私有。单seed、已查看开发域、sharedHDR与历史seed缺失不支持SD、显著性、新盲测、普适性或因果保证；旧GeoPRR历史pixel／完整权重链未重认证，本次未加载checkpoint／RNG内容。四阶段complete／exit0，保存检查时父进程退出／GPU空闲。全部旧资产保留，无自动promote。
