# V30：因子化二阶语义标量读出

**源85轮／best65及两域各一次冻结外评已完成；唯一保存数字／执行核对71441项PASS、0失败，不升级默认。** [完整报告](completed/report_cn.md)、[比较表](completed/comparison.csv)、[指标](completed/metrics.json)与[核对记录](completed/numeric_execution_audit.json)保留混合结果。Industrial mean15.418735／tail56.544002／max80.651760%FS，较V24／V25／V26／V29三项均改善，但仍弱于V28；较v16的mean／tail更高、max更低。RF100 mean12.095891%FS弱于主要参照，tail50.749786／max68.491405%FS低于V26／V28／V29；相对V24 >5救回37／新增20，同时mean／tail升高。Industrial相对V26 >5救回127／新增147，mean／tail改善未覆盖全部阈值。

[协议](../../GEOPRR_V2_BILINEAR_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)与[配置](config.json)记录完整fresh cached ImageNet MobileNetV4、GAP1／semantic1280 FP16及新FP32二阶point：Linear1280→128含bias、分为A／B各64项、sum(A·B)/sqrt64＋scalar bias，再FP32 sigmoid。实际8598481参数／234个trainable参数tensor／465state；encoder8434512参数／462state初始化保留，point163969参数／3tensor／3state，product stage无参数／state。A std1/sqrt1280、B std.001、factor／output bias0固定。全部CNN／head源训练，BN仅source更新／eval冻结；没有旧linear point、V28 pool5120、V29 style或V27固定BN。精确[readout contract](completed/protocol.json)已保留。

权威输入保持RGB512／output128／原ROI／标签／photo／roster，旧256只为population provenance。完整源R选best65，85−65=20按min60max160停止，LR下降不重置counter，无resume。last85只源；两域在完整源停止后各一次冻结同一best，目标训练／统计拟合／BN适配／checkpoint或超参选择均0。两源best／last各原1506＋roll512 CSV完整保留，roll512只自身报告；V27无外域结果，不作为baseline。

正式CNN／point各3145866图／49240调用，46495次源backward尝试＝46475更新＋20AMPskip。Point-only CUDA-event forward2.315521071秒已包含在full model／pipeline内，不含head backward；pref3CNN／8图／1back／0optimizer与point0.052910719秒单列。训练pipeline14485.251577秒、source full15146.585475秒、source peak allocated／reserved8266190336／11351883776bytes；[实际成本](completed/costs.json)保留全部边界，一次CNN不等于同成本。

[completed allowlist](completed/public_allowlist.json)26项，[总allowlist](public_allowlist.json)33项。完整scalar CSV、paired counts、scalar group summaries、fixed cohorts与必要focus／maxIDs保留；rawrun、fullcontracts、完整人口／fixed-cohort名单、per-image delta maps、完整improved／regressed与threshold／rescue／new名单、RGB、weights、latent features、geometry、bbox及privatepaths私有。单seed、已查看开发域、sharedHDR与历史seed缺失不支持SD、显著性、新盲测、普适性或因果保证；旧GeoPRR历史pixel／完整权重链未重认证，本次未加载checkpoint／RNG内容。四阶段complete／exit0，保存检查时父进程退出／GPU空闲。全部旧资产保留，无自动promote。
