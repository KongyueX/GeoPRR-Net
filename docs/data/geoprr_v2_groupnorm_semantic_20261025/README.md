# V31：单图分组归一化语义读出

**V31源预测坍缩为近常数，第13至60轮未恢复有效源读数；完整60轮／best1及两域各一次冻结外评已结束，不升级默认。** 唯一保存数字／执行核对68416项PASS、0失败。[完整报告](completed/report_cn.md)、[指标](completed/metrics.json)、[比较表](completed/comparison.csv)与[核对记录](completed/numeric_execution_audit.json)保留负结果。best源原1506／roll512 mean25.159314／24.600501%FS，预测范围均为[0.5042189359664917,0.504226565361023]，std分别1.9063593963920542e-6／1.8330245393135286e-6。该保存分布支持近常数输出观察，不确定精确根因。

Industrial mean37.249295／tail50.422275／max50.422275%FS，Acc@1／2／5全0%；RF100 mean23.395320／tail49.616268／max50.422275%FS、Acc@5=9.271523%。相对v16 >5救回／新增为Industrial0／626、RF6／85；相对旧GeoPRR为0／641、3／94；相对V30为0／542、6／74。更低max或局部RF tail不能抵消mean、准确率与新增大错的整体退化。

[协议](../../GEOPRR_V2_GROUPNORM_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)与[配置](config.json)固定77个BN替换为8组GN、eps1e-5。实际8435793参数／233个trainable参数tensor／233state，GN77／BN0／buffer0；encoder8434512参数／231state，初始化保留231个learned cached tensor，明确移除231个原BN运行buffer，不宣称原462encoderstate全部保持。GAPsemantic1280和新Linear1280→1 point，全参数源训练；learned参数保持不等于预训练函数保持。

GN在train／eval均按当前单图固定公式计算group-channel／spatial统计，FP32数学后转回原feature FP16再drop→act，外层encoder／semantic／point FP16、scalar sigmoid FP32。eval单图统计正常计算，但不总体目标统计拟合或参数适配。完整源R选best1，min60使第60轮停止、best年龄59；LR下降不重置counter，无resume。两域在源完整停止后各一次冻结同一best1，last60只源；roll512只自身报告，不补评旧模型。

正式CNN／point各2221066图／34765调用，32820次源backward尝试＝32809更新＋11AMPskip。pref3CNN／8图／1back／0optimizer、233 strictreload及必要梯度见证单列；pref通过不是学习成功。Point／GN-only CUDA计时未记录，保持unknown／null而非0；[实际成本](completed/costs.json)保留完整pipeline／process／peak。训练pipeline10325.968896秒、source full10805.576444秒，source peak allocated／reserved12174175232／13864271872bytes。

[completed allowlist](completed/public_allowlist.json)26项，[总allowlist](public_allowlist.json)33项。完整scalar CSV、paired counts、scalar group summaries、fixed cohorts与必要focus／max示例保留；并列max仅展示原保存第一条ID，max数值不变、完整逐图CSV保留。rawrun、fullcontracts、完整人口／cohort ID数组、delta maps、完整阈值救回／新增及长max并列名单、weights、RGB、features、瞬时统计值、geometry与privatepaths私有。单seed、已查看开发域、sharedHDR与历史seed缺失不支持SD、显著性、新盲测、普适性或因果；旧GeoPRR历史pixels／完整权重链未重认证，checkpoint／RNG内容未加载。四阶段complete／exit0、保存检查时父子退出／GPU空闲，全部旧资产保留，无自动promote。
