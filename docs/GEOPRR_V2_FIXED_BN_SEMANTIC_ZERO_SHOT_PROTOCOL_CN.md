# 固定预训练BN统计的完整语义适配：zero-shot协议

2026-10-25。V27检验完整可训练语义表示在固定ImageNet归一化统计下的源域适配。保持V26的native ROI RGB512、完整MobileNetV4 semantic1280和新scalar point；全部CNN、BN affine与point参与源训练，77个BN只使用原缓存ImageNet running statistics。没有新增图像、额外分支或旧task初始化。

## 问题与单一变化

V26源best89的原1506 mean／tail10／max为0.329670／1.376887／17.664927%FS，roll512为0.504289／2.118687／11.576979%FS；best R为1.082383%FS，last109为1.098346%FS。训练NMAE同期从0.252844降至0.196736%FS。源拟合仍在改善，源验证风险未同步下降且保留大错；这些保存标量支持另检表示适配约束，不证明BN运行统计是成因。

已有源训练在低平均误差下仍保留严重错误，且原验证与roll验证组成不同。全网络源训练同时改变卷积、BN affine及BN运行统计，不能从源R下降推断预训练统计仍适合当前表示。V27固定运行统计，保留卷积与affine适配能力，检验这一归一化表示约束是否改善源尾部和冻结外域表现。方向根据源域证据与模型接口确定，不根据Industrial／RF结果或具体病例选择。

这与V24的完整encoder参数固定、外接adapter不同：V27全部原CNN与BN affine可训练，没有adapter。与V26相比只改变BN训练时使用和保留运行统计的策略；不改512输入、source人口、photo、scalar loss、源R与停止规则。不承诺固定统计保持训练后的特征语义、输出不变或泛化，也不据此认证“语义遗忘”因果。

## 模型与归一化合同

完整`mobilenetv4_conv_medium.e500_r224_in1k`、`num_classes=0`，仅既有缓存ImageNet权重，保留conv_head／norm_head、8434512 encoder参数／462 state和实际semantic1280。reported `num_features=960`不替代实际语义维度。新Linear1280→1，weight Normal(std=.001)、bias0；logit转FP32后sigmoid，唯一scalar。预期8435793参数／233个参数tensor／464 state，实际构造与runtime记录为准。

模型只接受image。源训练时外层与非BN encoder模块保持training；所有77 BN强制eval，使用原ImageNet running_mean／running_var，num_batches_tracked不变。BN的weight／bias仍需梯度，和全部CNN、point一起进入AdamW与clip。不能把原参数固定、整个encoder eval或no_grad当作本轮实现。

77个BN的231个buffer（mean／variance／counter）须与fresh缓存ImageNet参考逐项一致。每轮在设备上直接比较全部231个buffer；checkpoint保存原始参考，resume沿用并严格检查，不能把恢复时当前值重新指定为原始参考。该参考不加入部署state，不增加464模型state，不公开其数值。BN affine不属于冻结统计。

## RGB与源人口

`native ROI → OpenCV LINEAR resize512 → RGB／255 → ImageNet normalize → 单CNN → semantic1280 → point → reading`。配置image_size512／output_size128／require_geometryFalse／pairNone；128仅满足dataset API，不生成或认证几何场。失败占位3×512×512。没有新增目标crop、GT几何、针尖、中心、H或router。

完全复用V22完成的22686 unique：旧18590每轮一次＋4096 roll每轮四次=34974呈现／547批64，末批30。原验证1506／24调用＋roll512／8，共2018／32。四manifest、行顺序、scalar标签、原ROI及contrast／saturation .8–1.2、brightness ±.1不变；不渲染、不加增强、不重查旧GT。保留V26权威RGB512合同；旧256说明仅population provenance，不能称RGB流程等于v16／v25。

roll512独立layout／ID／group／声明generated seed，但共享可用HDR；原20096 metadata缺seed，不认证全历史seed链。原1506 scene guard、原14494／1506／4000、ordinary1495／1497／severe6及固定focus不变。不对旧模型补评新roll或512。

## 源优化、选择与停止

唯一scalar SmoothL1(beta=.05)。fresh AdamW LR3e-4／wd1e-4／clip5／AMP／default fresh GradScaler／seed20262020／train64／val64／workers4。全部参数可训练，BN buffer固定。encoder／point FP16，scalar FP32；不扫描精度、scale、seed、width、分辨率、R权重或新模块。

R=`(mean_original+mean_roll+tail10_original+tail10_roll)/4`，normalized full error；input／scalar失败error1，tail取最大ceil10%，原151／roll52。最早严格最小R选best，同一R驱动Plateau factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8。完成epoch≥60且epoch−best_epoch≥20停止，否则最多160。counter从1计、LR下降和resume不重置，不延长。

完整model／optimizer／scheduler／scaler／RNG／history／totals／initialization、新protocol／config／源合同及原BN参考保存并严格恢复。BN策略不符、原参考不符或V26旧protocol不能恢复为V27。last只保留两源记录。训练量按本轮冻结规则实际执行，不强凑旧109轮或同预算。

## 必要新检查与串行执行

仅针对新BN策略、梯度和恢复开展CPU toy检查，真实model CPU构造0forward／0cachedweights／0RGB／0GPU。检查train()后77BN eval、其他CNN train、affine与CNN／point梯度、optimizer实际可更新参数而buffer固定、strict resume原参考与策略。旧dataset、GT、数学、教师、G0及旧tests不重跑。

新真实源AMP零更新见证恰3CNN／8图／1combined backward／0optimizer：4张既有roll train0／1＋两源验证各2。实际512输入／semantic1280／F16 encoder与point／FP32 scalar、所有462 cached encoder初始化逐项保留、全部233参数无optimizer不变、CNN／stem／完整neck／BN affine／point梯度有限非零、231 BN buffer源训练与eval逐项不变、77 counter源+0／eval额外0、464 strict reload、两验证512 LINEAR／ImageNet identity均记录。pref固定loss×128后grad÷128，正式default fresh GradScaler不改，正式fresh重建不沿用pref状态。

V26整阶段完成、ONE独立数字执行核对并归档、原父子退出／GPU空闲、本轮代码和配置冻结后，才启动唯一pref→source→Industrial→RF串行队列。活job只监控，不改livecode／模型／输入／BN策略／名单／标签／loss；4workers不是额外模型，阶段交接短暂空闲不能启动第二GPU任务。

## 冻结外评与成本

source按冻结规则完整停止后，同一source-best对Industrial1395／52与RF100151／35各一次冻结评估；last不外评。两个已查看开发域不参与训练、总体统计拟合、BN适配、checkpoint／epoch／早停／LR／训练超参／ROI／阈值／温度选择，不根据目标病例改标签。forward仅nativeROI RGB，RFscalar离线四点派生；失败保留完整分母error1，不删行或误判有效饱和预测。

E轮train34974E图／547E CNN及loss backward尝试，updates＋AMPskip=547E；validation2018E／32E；target1546／25／0back／0update。正式总36992E＋1546图／579E＋25调用，point hooks与CNN逐项相同。原CNN仍反传，冻结buffer不是免费计算；实记BN检查／梯度／calls／updates／skips／time／peak／init／resume／R／LR／stop，pref单列，旧源渲染不重复计费。

## 完整核对、归档与继续研究

整阶段四exit0、父子退出／GPU空闲后，ONE新鲜独立CPU保存标量和执行核对；实现作者不能充当独立auditor，不再起第二审计或旧GT／模型评估。收齐两源best／last1506＋512、完整I/R逐图／组／summary、计数／初始化／BN固定证据／时间与源选择目标隔离。公开显式规范化allowlist与task code至现有main，禁止新branch／gitadd-A。

完整报告mean／Acc1/2/5／median/p90/p95／tail10／max、与v16／旧GeoPRR／v24／v25／v26和保存参照的>1/2/5新增救回、逐图与组改善恶化、zero／nonzero／nearzero、ordinary1495／1497／severe6及固定focus。两域均值、尾部、max和新增大错共同判断，源R或单域收益不能代表整体成功，不自动升级模型。所有旧资产与正负结果保留，归档后继续独立源域结构／表示研究。

## 实际结果：源训练失败

V27真实源AMP零更新见证通过（3CNN／8图／1combinedback／0optimizer；231固定BN buffer、154 affine gradient tensors有限非零、464 strict reload）。正式首批64图loss0.21415004134178162、LR3e-4、gradient norm finite；随后在epoch1 batch36因nonfinite scalar loss失败，2026-10-24T21:05:28.692328+00:00源阶段exit1。完成源epoch为0，没有history／latest／best／completed checkpoint，Industrial／RF100均未启动，没有外域结果。保存检查时parent／trainer已退出、GPU空闲。

实际optimizer updates与AMP skips总计未知，记null而非0。代码只推出失败前35次完整backward尝试与第36批64图forward失败；该partial code-implied信息不是保存counter观测总量或完整正式预算。根因尚未确定，不认定FP16 overflow。全部资产保留，不以同配方重启、不改变LR／precision或放松BN统计保护，不自动升级模型。[失败报告](data/geoprr_v2_fixed_bn_semantic_20261025/report_cn.md)、[失败记录](data/geoprr_v2_fixed_bn_semantic_20261025/failure_record.json)与[9项显式allowlist](data/geoprr_v2_fixed_bn_semantic_20261025/public_allowlist.json)记录实际结果；本轮不产生完整阶段数字审计或外域比较。

## Limitations

固定BN运行统计约束归一化通路，卷积与affine仍变化；不能认证预训练语义保持、错误成因或必然泛化。只运行单seed，两个已查看开发域不是新盲测，不报告SD、显著性、普适性或新理论。共享HDR源验证与历史seed缺失边界保留，旧GeoPRR历史pixel／完整权重链未重新认证。RGB／weights／BN参考值／H/C/XYZ/U/M／bbox／features／rawrun／fullcontracts／完整源数组／privatepaths留私有，仅公开规范化标量与必要执行metadata。
