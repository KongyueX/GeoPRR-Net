# 2×2空间语义直接读数：zero-shot协议

2026-10-25。V28将完整缓存ImageNet MobileNetV4的最终平均池化由1×1改为固定2×2，保留原conv_head／norm_head对四个空间cell的1280维语义变换；按原NCHW flatten顺序得到5120维表示，接唯一新Linear5120→1／FP32 sigmoid。全部CNN、BN affine和point源训练，BN只在source train更新、eval冻结。没有新增RGB、foundation、teacher、几何分支、query、融合或TTA。

## 源依据与结构变化

V26源best89的原1506 mean／tail10／max为0.329670／1.376887／17.664927%FS，roll512为0.504289／2.118687／11.576979%FS。低平均误差仍伴随大错。原1×1平均池化不保留最终特征的粗空间排列；下一实验只检验一个最小空间语义表示，不把源错误直接归因为空间信息丢失。

固定2×2是本轮唯一pool grid，不扫描grid、分辨率、seed、读出宽度或权重。选择依据为保存源标量与实际安装timm的池化／neck接口，不使用外域病例、分数或标签定grid。V27固定BN统计的源训练在epoch1 batch36因非有限scalar loss终止；它没有完成epoch或外域结果，保持失败记录，不通过降低LR、改精度或放宽保护续跑。V28为独立结构实验，BN沿用V26的source更新策略，不声称已确定V27失稳成因。

## 单CNN与完整neck

使用既有缓存`mobilenetv4_conv_medium.e500_r224_in1k`、num_classes0，原8434512 encoder参数／462 state、conv_head与norm_head保留。原global_pool替换为无参数的adaptive average2×2，执行顺序仍是`forward_features → pool2×2 → conv_head → norm_head → act2 → 原flatten`，不提前移动或移除neck。

native ROI RGB512时最后特征960×16×16，pool后960×2×2，neck后1280×2×2，flatten5120。NCHW flatten按channel、row、column展开，每channel内依次左上／右上／左下／右下；不将其误写为四个连续1280向量拼接，也不认证针尖／起终点角色。新point weight Normal(std=.001)、bias0，唯一scalar；不构造或融合旧1280 point。预期新point5121参数、总8439633参数／233参数tensor／464 state，实际CPU／runtime确认。

完整462 cached encoder state在初始化逐项保留；无参数pool变化不保证初始化输出等于旧GAP模型，也不保证ImageNet语义不变。全部原CNN／BN affine／point可训练，BN运行统计只source更新。没有V27的固定BN参考或全encoder eval／no_grad。

## 输入、人口与源学习

完全复用V26的权威RGB512／output128／原nativeROI／LINEAR／ImageNet normalize／contrast,saturation .8–1.2／brightness±.1。128仅dataset接口，无geometry target或pair manifest；失败占位3×512×512。旧256说明仅population provenance，不称输入等于v16／v25。

source22686 unique，旧18590每轮一次＋4096 roll四次=34974呈现／547批64末30；原val1506／24 calls＋roll512／8=2018／32。四manifest的rows／顺序／标签／ROI保持，无新RGB／增强／GTcrop／源seed／渲染／旧GT检查。roll512独立layout／ID／group／declaredseed但共享HDR，原20096metadata无seed不认证历史全链；原1506scene guard及14494／1506／4000、fixed ordinary1495／1497／severe6和focus保持。

唯一scalar SmoothL1(beta=.05)，fresh AdamW LR3e-4／wd1e-4／clip5／AMP default fresh GradScaler／seed20262020／train64val64workers4。encoder与point FP16、scalar FP32；不扫描精度或scale。

源R=(原mean＋rollmean＋原tail10＋rolltail10)/4，normalized fullerror，失败1，tail最大ceil10%为151／52。earliest strict min R选best，同R驱动Plateau factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8。min60／max160，完整epoch≥60且epoch−best_epoch≥20停止，counter从1，LR／resume不重置，不延长旧109轮或强凑同预算。完整model／optimizer／scheduler／scaler／RNG／history／totals／init／新config／protocol／源合同严格保存恢复，拒旧head或旧protocol；last只源。

## 必要新检查与串行执行

新增CPU toy仅检查2×2空间顺序、相同GAP而不同空间分布的可区分性、完整neck四cell与5120 scalar读出、单RGB接口／梯度及新head严格恢复。真实model CPU只0forward／0cachedweights／0RGB／0GPU构造确认实际metadata；不重跑旧数据／R数学／GT／教师／G0／旧tests。

一次真实源AMP零更新见证恰3CNN／8图／1combinedback／0optimizer：4张既有rolltrain0／1＋两源验证各2。记录实际512输入、960×16×16→960×2×2→1280×2×2→5120→1的真实shape、FP16 latent／point与FP32scalar、462cached初始化逐项保持、encoderstem／neck／point有限非零梯度、全部233参数无optimizer不变、77BN源+1／eval额外0、464strictreload及两val512identity。pref固定loss128再grad/128，formal fresh default GradScaler不改，正式fresh重建不沿用prefBN。

V26已完整ONE核对归档，V27失败资产已保留且原父子退出、GPU实际空闲、本轮code/config冻结后，才启动唯一pref→source→Industrial→RF串行队列。不能要求V27完成或盲恢复其失败任务。活job只监控queue／currentstep原log／history/latest／实际父子GPU，不改livecode／model／config／名单／标签／loss／poolgrid／输入／BN策略，不增加GPU或提前评估，4workers不是额外模型。

## 最终评估、成本与归档

source按规则完整停止后，同一source-best只对Industrial1395／52与RF100151／35各一次冻结评估；last不外评。两已查看开发域不训练、总体统计拟合、BN适配、checkpoint／epoch／早停／LR／训练超参／grid／ROI／阈值／温度选择，不用目标病例设计标签。forward只有给定nativeROI RGB；RFscalar离线四点派生，input/scalarfail保留完整分母error1，不删行或把有效饱和值误叫invalid。

E轮实际train34974E图／547E CNN和backwardattempt=updates＋AMPskip，val2018E／32E；targets1546／25／0back／0update，总36992E＋1546图／579E＋25 calls，point hooks一致。四cell neck／5120point虽单CNN但不是免费，实记calls／shape／梯度／updates／skip／time／peak／init／stop／resume／R／LR与源选择目标时间隔离；pref单列，旧渲染不重复计费。

whole source＋I/R complete、四exit0、父子退出／GPU空闲后，才委派ONE新鲜独立CPU保存数字执行核对，实现作者不充当独立auditor、不并行第二审计。收齐两源best/latest逐图、I/R全部scalar/group/summary、protocol/config/runmetadata/history/completed和实际成本，规范化explicit allowlist与task code推existing main，不branch／gitadd-A。

报告完整mean／Acc1/2/5／median/p90/p95／tail10／max、相对v16／旧GeoPRR／v24／v25／v26及保存参照的>1/2/5新增救回／逐图组改善恶化／zero非zero近zero／ordinary1495/1497/severe6／固定focus；roll512独立报告，不补评旧模型。V27没有外域结果，不能拼造参照。均值、尾部及新增大错共同判断，不源R或单域收益即称整体成功，不自动promotion；保留全部旧正负资产，归档后继续独立源域结构／表示研究。

## Limitations

2×2只保留粗排列，不恢复更早特征或精细坐标，不保证旋转不变、几何正确、语义保持或外域收益。pool、point维度、BN空间样本与训练历史同时变化，不作单因素因果认证。单seed、已查看开发域、sharedHDR与历史seed缺失边界保持，不报SD、显著性、新盲测、普适性或新理论；旧GeoPRR历史pixel/fullweight链未重认证。RGB／weights／H/C/XYZ/U/M／bbox／features／rawrun／fullcontracts／源arrays／privatepaths私有，只公开必要规范化标量和执行metadata，不买算力／新foundation／新权重或额外seed，不碰officialtest4000或其他项目，不第三方消息。
