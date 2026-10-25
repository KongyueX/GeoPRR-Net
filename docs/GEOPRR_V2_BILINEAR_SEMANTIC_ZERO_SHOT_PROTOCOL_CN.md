# 因子化二阶语义标量读出：zero-shot协议

2026-10-25。V30保留完整fresh缓存ImageNet MobileNetV4、原1×1平均池化及semantic1280，将最终线性point换成一个64项因子化乘积读出。CNN仍为单RGB通路，全部CNN与新head源训练；BN仅source更新、eval冻结。没有源feature style混合、几何分支、teacher、旧point残差、标量融合或TTA。

## 源依据与单一问题

V29源best15原1506 mean／tail10／max为0.749396／3.505846／82.507033%FS，roll512为0.910155／3.273653／13.418275%FS，R=2.109762%FS。源R在15轮后未取得严格更小值，按min60约束完整停止。该固定style变体未改善原生源验证，不能据此提前宣称其外域结果或确定错误成因。

V26原完整GAP语义标量的源best89 mean／tail为原0.329670／1.376887、roll0.504289／2.118687%FS，仍保留较大源错误。V30以这一较简单源通路为基础，检验单线性读出以外的语义channel乘性交互；不再改变早期特征统计，不扫描V29的p、α或eps来救回结果。

本轮方向、64项容量、初始化、FP32新head与预算在root读取V29外域结果前固定。低秩因子只是读出参数化，不给latent指定针尖／起终点角色，也不证明源错误由线性容量引起或保证泛化。

## 单标量head与初始化

原完整`mobilenetv4_conv_medium.e500_r224_in1k`、num_classes0，既有缓存ImageNet权重，conv_head／norm_head、8434512 encoder参数／462 state与实际semantic1280保留；reported num_features960不是读出维度。仅原GAP1，没有V28的pool2×2或V27固定BN策略。

对完整语义向量h，单Linear1280→128含bias，按前64／后64分为a(h)、b(h)。唯一logit：

`z = sum_k a_k(h)*b_k(h) / sqrt(64) + c`，`reading = sigmoid(z)`。

这是一组64项乘积，不将对称二阶矩阵的秩误称为严格64，也不是两个读数融合。factor bias允许隐式线性／常数项，无独立旧point分支、router或单case gate。1/√64为固定维度归一化，不拟合目标温度。

factor weight前64行Normal(std=1/√1280)，后64行Normal(std=.001)，128个factor bias与output bias c均0。该固定非对称初始化使一个因子保持尺度、另一个因子保持小初始logit；不扫描初始化、rank或width。没有旧task权重或旧point构造后丢弃。

encoder在原FP16 autocast中执行；新head明确关闭autocast，对semantic.float()作FP32 Linear、乘积、sum／尺度与bias，logit和sigmoid均FP32。这是新乘法算子的固定数值实现，不做FP16／FP32比较或precision／GradScaler扫描。全部CNN和head参数共同训练，原BN运行统计仅源更新、eval冻结。

预期head163969参数（Linear weight163840＋bias128＋output bias1）、3个参数tensor／3 state；完整8598481参数／234个参数tensor／465 state。必要stateless乘积stage仅用于真实shape／dtype见证，不新增参数、buffer、feature缓存或开关。实际CPU构造及runtime记录为准。

## RGB、人口与源优化

完全复用V22完成22686unique、V26 RGB512／output128／nativeROI／LINEAR／ImageNet normalize／原contrast,saturation .8–1.2、brightness±.1。旧18590每epoch一次＋roll4096四次=34974呈现／547批64末30；原val1506／24＋roll512／8=2018／32。四manifest rows／顺序／标签／ROI不变，无新RGB／增强文件／源seed／GTcrop／渲染或旧GT检查。128只是dataset接口，无geometry target／pair；旧256合同仅population provenance。

保留原1506scene guard、14494／1506／4000、fixed ordinary1495／1497／severe6和focus。roll512独立layout／ID／group／declaredseed但sharedHDR；原20096metadata缺seed，不认证全历史链，不补评旧模型新roll或新分辨率。

唯一scalar SmoothL1(beta=.05)，freshAdamW3e-4／wd1e-4／clip5／AMP default fresh GradScaler／seed20262020／train64val64workers4。无额外loss、teacher、style、label mix、target feature统计拟合或校准。新head与初始化单一冻结，不加seed／宽度／权重／精度矩阵。

R=(原mean＋rollmean＋原tail10＋rolltail10)/4，normalized fullerror，input／scalar失败1，tail最大ceil10%为151／52。最早strict minimum R选best，同R驱动Plateau factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8。min60max160，完成epoch≥60且epoch−best_epoch≥20停止，counter从1、LR／resume不重置，不强凑旧轮数或延长。

完整model／optimizer／scheduler／scaler／RNG／history／totals／init／新protocolconfig／源合同保存并严格恢复，旧head形状或旧protocol不能恢复为V30；last仅源。

## 必要新检查与串行执行

只新增CPU toy检查64项公式／channel交互、FP32 head在autocast下的真实dtype、两因子与encoderstem／neck梯度、B1／B3、encoder初始化不变／无unused point、465 strict reload与拒旧protocol／形状。真实model CPU只0forward／0cachedweights／0RGB／0GPU构造确认metadata；旧R数学、GT、教师、G0、旧tests或模型不重跑。

一次真实源AMP零更新见证恰3CNN／8图／1combinedback／0optimizer，4既有rolltrain0／1＋两源验证各2。记录RGB512→GAP960×1×1→完整semantic1280 FP16→projection128／两半64／product64 FP32→logit1／scalar FP32；cached462初始逐项保持、234参数无optimizer不变、encoderstem／完整neck及两factor weight区块梯度有限非零、77BN源+1／eval额外0、465 strict reload、两val512 LINEAR／ImageNet identity。output bias梯度可因样本抵消为0，不把这一数学可能当故障；必要参数梯度必须存在且有限。pref固定loss128再grad/128，正式fresh重建default GradScaler不改、不沿用prefBN。

V29整个source＋I/R complete、ONE新鲜CPU数字执行核对与规范化归档、原父子退出／GPU实际空闲、本轮code/config冻结后，才启动唯一pref→source→Industrial→RF串行队列。活job只监控pipeline.pid／queue/currentstep原log／history/latest与实际父子GPU，不改livecode／模型／配置／名单／标签／loss／rank／初始化／precision／BN／ROI，不加GPU或提前评估。4workers不是额外模型，best不更新不是停滞，正常AMPskip如实保留。

## 冻结外评、成本与归档

source按冻结规则完整停止后，同一source-best对Industrial1395／52与RF100151／35各一次冻结评估；last不外评。两已查看开发域不训练、总体统计拟合、BNadapt、checkpoint／epoch／早停／LR／训练超参／rank／初始化／阈值／ROI／温度选择，不用目标病例设计标签。forward只给定nativeROI RGB，RFscalar离线四点派生，input／scalar失败完整分母error1，不删行或误判有效饱和值。

E轮train34974E图／547E CNN及backattempt=updates＋AMPskip，val2018E／32E；target1546／25／0back／0update，总36992E＋1546图／579E＋25calls，point与factor projection实际调用／图数须与CNN相同。FP32 factor projection与乘法虽单CNN但不是免费；记录head forward事件耗时、完整pipeline／peak／counts／init／stop／resume／R／LR／source-target隔离，pref单列，不重复计旧渲染。

whole source＋I/R complete、四exit0、父子退出GPU空后才委派ONE新鲜独立CPU保存数字／执行核对，实现作者不能担任、不并行第二审计。收齐两源best/latest逐图、全部I/R scalar/group/summary、protocol/config/runmetadata/history/completed及实际成本；仅规范化explicit allowlist与taskcode推existing main，不branch／gitadd-A。

报告mean／Acc1/2/5／median/p90/p95／tail10／max、相对v16／旧GeoPRR／v24／v25／v26／v28／v29及保存参照的>1/2/5新增救回／逐图组改善恶化／zero非zero近zero／ordinary1495/1497/severe6／固定focus，roll512只自身，V27无外域结果不拼造。源R或单域收益不代表整体成功，不自动promotion；旧资产、G1／anchor／65轮／配对守恒负结果全保留，归档后继续独立源域结构／表示研究。

## Limitations

因子化乘积只增加学习到的语义交互，不能认证真实几何角色、相对读数因果或尾部收益。Head参数化、初始化、FP32计算与训练历史同时变化，不作单因素因果主张。单seed、已查看开发域、sharedHDR与历史seed缺失边界保持，不报SD、显著性、新盲测、普适性或新理论；旧GeoPRR历史pixel/fullweight链未重认证。RGB／weights／features／H/C/XYZ/U/M／bbox／rawrun／fullcontracts／源arrays／privatepaths保持私有，只公开必要规范化标量、shape／dtype和执行metadata。
