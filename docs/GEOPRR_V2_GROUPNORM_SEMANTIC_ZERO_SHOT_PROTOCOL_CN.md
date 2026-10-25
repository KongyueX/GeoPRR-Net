# 单图分组归一化语义读出：zero-shot协议

2026-10-25。V31将完整缓存ImageNet MobileNetV4内77个BatchNormAct2d替换为固定8组GroupNorm，复制原学习到的affine参数并保留drop→act顺序。原1×1平均池化、1280维neck与新Linear1280→1／FP32sigmoid保持，全部CNN、norm affine与point源训练。没有running statistics、早期style混合、二阶head、teacher、几何分支、融合或TTA。

## 源依据与固定问题

V30源best65的原1506 mean／tail10／max为0.560196／2.468517／20.804057%FS，roll512为0.975716／4.003882／31.504238%FS，R=2.002078%FS。V29源best15 R=2.109762%FS，原源max82.507033%FS。两次实际历史不同，不能由此断定二阶读出或style变体的外域结果或错误成因。

V26完整GAP线性语义通路的源best89 R为1.082383%FS，仍有大错。V31回到这一简单读出，检验不依赖batch人口或EMA运行统计的内部表示。源训练混合原／roll人口，训练与原生验证的表示条件不同，支持单独检验归一化约束，不证明BN造成错误。

方向、8组、保留原eps、FP32 norm运算／原dtype输出、源优化及预算在root读取V30外域指标前冻结；不根据目标分数／病例定groups、eps、precision、ROI或训练参数。只运行这一项GroupNorm配置，不扫组数／层数／BN混合比例。

## 转换、初始化与新state合同

先构建既有缓存`mobilenetv4_conv_medium.e500_r224_in1k`、num_classes0，再转换77个归一化节点。原完整encoder含8434512学习参数／231参数tensor／462 state，其中231 state是BN running_mean／running_var／num_batches_tracked。V31保留全部231个学习参数的值、名称与形状，明确丢弃231个BN运行buffer；不能声称原462 encoder state全部保持。

原安装架构的channel集合为32／48／80／128／160／192／256／320／480／512／640／960／1024／1280，全部可被8整除；77层eps均为1e-5，原drop均Identity，act为Identity或ReLU。这是一次CPU架构元数据读取，0forward／0cachedweights／0RGB／0GPU，不是新准确率结果或旧模型补评。

GroupNormAct直接拥有weight／bias，保持原parameter key；保留原drop与act对象／顺序，不将旧BN注册为子模块，不保留原running buffer或统计bank。Group数固定8，eps逐层取原1e-5。GN数学运算在disabled autocast中对feature.float()与FP32 affine执行，归一化结果先转回输入dtype，再执行原drop→act。必要stateless math stage仅供真实dtype／shape见证，0参数／buffer／state，不缓存特征或瞬时统计。

GN在每张图内部按channel组与空间位置计算瞬时均值／方差，训练和eval使用同一公式，样本间不聚合统计。这里的推理期单图归一化属于冻结网络forward，不在目标人口上拟合统计、不积累状态、不更新affine，也不是BN adaptation。不得把“无running statistics”写成“forward不计算任何单图统计”。

完整conv权重与conv_head学习参数保持，norm_head的channel、affine与激活保留，但归一化函数已经改变；不认证原ImageNet输出或语义保持。新point1280→1 weight Normal(std=.001)、bias0，logit转FP32后sigmoid唯一scalar。预期总8435793学习参数／233参数tensor／233 state（encoder231＋point2）、GN77／BN0／running buffer0；实际CPU和runtime确认。模型只接受image，全部学习参数可训练。

## 数据与源优化

完全复用V22完成22686unique、V26权威RGB512／output128／nativeROI／LINEAR／ImageNet normalize／contrast,saturation .8–1.2／brightness±.1。旧18590每epoch一次＋roll4096四次=34974呈现／547批64末30；验证原1506／24＋roll512／8=2018／32。四manifest rows／顺序／scalar标签／ROI不变；无新RGB／增强／源seed／GTcrop／渲染／旧GT检查。128只是dataset接口，无geometry target／pair，旧256合同仅population provenance。

保留原1506scene guard、14494／1506／4000与fixed ordinary1495／1497／severe6及focus。roll512独立layout／ID／group／declaredseed但sharedHDR，原20096metadata缺seed不认证全历史链。不对旧模型补评新roll、GN或分辨率。

唯一scalar SmoothL1(beta=.05)，freshAdamW3e-4／wd1e-4／clip5／AMP default fresh GradScaler／seed20262020／train64val64workers4。encoder／point采用原FP16外层autocast，GN内部FP32后恢复原dtype，scalar FP32；不比较其他精度、scale或新增seed。

R=(原mean＋rollmean＋原tail10＋rolltail10)/4，normalized fullerror，失败1，tail最大ceil10%为151／52。最早strict minimum R选best，同R驱动Plateau factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8及停止。min60max160，完成epoch≥60且epoch−best_epoch≥20停止，counter从1、LR／resume不重置，不强凑旧轮数或延长。

完整model／optimizer／scheduler／scaler／RNG／history／totals／init／新protocolconfig／normalization contract／源合同保存并严格恢复；GN groups、eps、dtype／activation顺序也是合同，只有相同权重shape不足以认证相同归一化。旧BN或二阶head checkpoint不能恢复为V31。last仅两源。

## 必要新检查与执行

只新增CPU toy检查学习参数／affine复制、原drop→act保持、移除buffer、8组／原eps、单图GN参考公式／FP32→原dtype、batch／mode独立、B1／B3、CNNstem／neck／norm affine／point梯度、233 strict reload与拒旧norm合同／protocol。真实NEW GN model CPU构造0forward／0cachedweights／0RGB／0GPU，确认参数与state metadata；旧R数学、GT、教师、G0或旧tests不重跑。

一次真实源AMP零更新见证恰3CNN／8图／1combinedback／0optimizer，4既有rolltrain0／1＋原val2＋rollval2。实际GN77／BN0／buffer0／学习参数231初值逐项等于缓存／原BN231 buffer已移除，GN math真实FP32、原feature／semantic1280／point FP16、scalar FP32；所有233参数无optimizer不变、必要encoderstem／neck／norm affine／point梯度存在且有限非零、无running state更新、233 strict reload、两val512 identity。GN瞬时统计可随输入改变，不能把无buffer更新误作无forward计算。

pref固定loss128再grad/128；正式fresh重建defaultGradScaler不改，不沿用pref状态。V30整轮source与I/R complete、ONE新鲜CPU数字执行核对与规范化归档、原父子退出／GPU实际空闲、新code/config冻结后，才启动唯一pref→source→Industrial→RF串行。活job只监控，不改livecode／model／config／名单／标签／loss／groups／eps／dtype／ROI，不加GPU或提前评估；4workers不是额外模型，best不更新不是停滞，正常AMPskip如实保留。

## 冻结外评、成本和归档

source按冻结规则完整停止后，同一source-best对Industrial1395／52与RF100151／35各一次冻结评估；last不外评。两已查看开发域不训练、总体统计拟合、BNadapt、checkpoint／epoch／早停／LR／训练超参／groups／eps／precision／阈值／ROI／温度选择，不用目标病例设计标签。forward只给定nativeROI RGB，GN只执行冻结的单图公式，RFscalar离线四点派生；input／scalar失败完整分母error1，不删行或误判有效饱和值。

E轮train34974E图／547E CNN及backattempt=updates＋AMPskip，val2018E／32E；target1546／25／0back／0update，总36992E＋1546图／579E＋25calls，point调用／图数与CNN相同。FP32 GN与dtype转换可能增加compute／activation memory，不能因参数相同或无buffer声称同成本；实际norm见证、时间／peak／calls／updates／skip／init／stop／resume／R／LR／source-target隔离记录，pref单列，旧渲染不重计。

whole source＋I/R complete／四exit0／父子退出GPU空后才委派ONE新鲜独立CPU保存数字执行核对，实现作者不担任，不并行第二审计。收齐两源best/latest逐图、全部I/R scalar/group/summary、protocol/config/norm contract/runmetadata/history/completed和实际成本；规范化explicit allowlist与taskcode推existing main，不branch／gitadd-A。

报告mean／Acc1/2/5／median/p90/p95／tail10／max、相对v16／旧GeoPRR／v24／v25／v26／v28／v29／v30及保存参照的>1/2/5新增救回／逐图组改善恶化／zero非zero近zero／ordinary1495/1497/severe6／固定focus。roll512只自身；V27无外域结果不拼造，源R或单域收益不代表整体成功，不自动promotion。旧资产、G1／anchor／65轮／配对守恒负结果全保留，归档后持续独立源域结构／表示研究。

## 完成记录

源60轮完整结束，earliest strict source R选best1，R=0.3624970918530252（36.249709185303%FS），last60 R=0.36415594591138356。60−1=59≥20，因min60在第60轮停止，未在第21轮提前结束；LR在10／19／28／37／46轮后下降不重置counter，无resume。同一best1只在源停止后各冻结外评一次，last只源；最终RF于2026-10-25T12:16:48.422736+00:00结束。四阶段complete／exit0，2026-10-25T12:22:51+00:00保存检查时父进程及各阶段进程全部退出、GPU空闲。

本轮保存源预测坍缩为近常数，第13至60轮未恢复有效读数。best源原1506／roll512 mean25.159314／24.600501%FS，预测范围均[0.5042189359664917,0.504226565361023]、std分别1.9063593963920542e-6／1.8330245393135286e-6；last roll保存输出std为0。第13／60轮train SmoothL1为0.24331026108168666／0.243289967727353，源R仍未低于第1轮。该负结果是保存scalar证据，不确定精确根因，不以pref通过替代学习效果。

Industrial mean37.249295／tail50.422275／max50.422275%FS，Acc@1／2／5全0%；RF100 mean23.395320／tail49.616268／max50.422275%FS、Acc@5=9.271523%。相对v16 >5救回／新增为Industrial0／626、RF6／85；相对旧GeoPRR为0／641、3／94；相对V30为0／542、6／74。更低max或局部RF tail不能写成整体成功；全部正负资产保留，不升级默认。

GN77／BN0／buffer0／233state，初始化只保留231个learned cached encoder tensor、移除231个原BN buffer；GN8的FP32数学转回原feature FP16后drop→act，eval正常计算单图统计但无目标人口统计拟合或参数适配。正式CNN／point各2221066图／34765调用，源backward尝试32820＝32809更新＋11AMPskip。pref3CNN／8图／1back／0optimizer及233 strictreload单列。Point／GN-only CUDA时间未记录，unknown／null而非0；完整pipeline／peak实记：训练10325.968896秒、source full10805.576444秒、peak allocated／reserved12174175232／13864271872bytes。

唯一独立CPU保存数字／执行核对68416项通过、0失败、最大差1.4210854715202004e-14；未加载模型／weights／RGB／features／统计值／geometry／checkpoint或重跑GT／旧测试／外评。[报告](data/geoprr_v2_groupnorm_semantic_20261025/completed/report_cn.md)、[指标](data/geoprr_v2_groupnorm_semantic_20261025/completed/metrics.json)、[比较表](data/geoprr_v2_groupnorm_semantic_20261025/completed/comparison.csv)和[26项completed allowlist](data/geoprr_v2_groupnorm_semantic_20261025/completed/public_allowlist.json)规范化归档；[总allowlist](data/geoprr_v2_groupnorm_semantic_20261025/public_allowlist.json)33项。并列max仅保留原保存第一条ID示例，不变更数值max，完整scalar CSV保留全部逐图记录；长并列名单、完整人口／cohort ID数组、delta maps、rawthreshold-rescue-new名单、rawrun／contracts、RGB／weights／features／瞬时统计值／geometry／privatepaths私有。checkpoint／RNG保存机制为代码与metadata证据，内容未加载，不增加恢复保证。

## Limitations

BN换为GN改变网络函数与预训练统计，保留学习参数不等于保持语义、输出或任务信息；单图归一化也可能移除有用信息。不能由源R差异确定BN成因或保证泛化，不作单因素因果主张。单seed、已查看开发域、sharedHDR及历史seed缺失边界保持，不报SD、显著性、新盲测、普适性或新理论；旧GeoPRR历史pixel/fullweight链未重认证。RGB／weights／features／瞬时统计实际值／H/C/XYZ/U/M／bbox／rawrun／fullcontracts／源arrays／privatepaths私有，仅公开必要规范化标量、shape／dtype和执行metadata。
