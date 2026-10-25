# 单早期层源特征风格混合：zero-shot协议

2026-10-25。V29在完整fresh缓存ImageNet MobileNetV4的一个早期层输出上，训练时混合当前源mini-batch的逐图channel均值与标准差；推理时该算子精确返回原特征，不计算目标统计。主通路为RGB512、原1×1平均池化、完整semantic1280、新Linear1280→1／FP32sigmoid。全部CNN与point源训练，BN仅source更新／eval冻结，无新增RGB、teacher、几何分支或融合。

## 源依据与固定方向

V28源best48的原1506 mean／tail10／max为0.358444／1.481549／21.598810%FS，roll512为0.566393／2.410069／15.727669%FS，R=1.204114%FS。V26源GAP best89的对应mean／tail为原0.329670／1.376887、roll0.504289／2.118687%FS，R=1.082383%FS。两次实际训练历史不同，不据此认证pool因果；源拟合与尾部支持本轮回到较简单的完整GAP语义通路，检验一个不同的表示正则约束。

现有photo限定亮度、对比和饱和度变化；roll验证共享可用HDR，不能替代风格独立性。V29混合早期源特征的逐图统计，以检验对特定源外观的依赖能否降低。方向、层位置、参数及预算在root读取V28外域结果前冻结，不根据目标分数或病例选择。

混合实例特征统计的思路参考[MixStyle原作者实现](https://github.com/KaiyangZhou/mixstyle-release)。本轮为单早期层、训练时始终启用、FP32 moments、population variance的固定变体；原实现默认p=.5，本轮p=1，α=.1与ε=1e-6固定，不作精确复现或新机制主张，原论文不构成本任务收益证据。

## 唯一源特征算子

插入点为`encoder.blocks[1]`输出，RGB512下实际应为B×80×64×64。只插入这一处，不扫层数、层位置、α、启用概率、eps、分辨率或seed。原完整encoder、1×1 global_pool、conv_head／norm_head、semantic1280和唯一scalar point保持；无V28的2×2 pool／5120 point、无V27的固定BN策略。fresh cached ImageNet encoder8434512参数／462 state初始化逐项保留；预期模型8435793参数／233参数tensor／464 state，实际CPU／runtime确认。

对当前源训练特征x，使用FP32逐图／逐channel空间统计：

`μ=mean_HW(x)`，`σ=sqrt(var_HW(x, unbiased=False)+1e-6)`。

μ与σ detach，内容`z=(float(x)−μ)/σ`仍对x反传。每张图采λ~Beta(.1,.1)，当前mini-batch随机排列π；`μmix=λμ+(1−λ)μπ`、`σmix=λσ+(1−λ)σπ`，输出`z·σmix+μmix`后转回原feature dtype。λ和π只用已seed的Torch全局设备RNG，checkpoint沿用原CPU／CUDA RNG保存恢复；不加独立model seed。self permutation允许，末批30按同规则处理，无domain名单或配对标签。

每个source train forward都启用该算子，没有概率gate、校准、confidence规则或target开关。eval在统计或随机操作前精确返回原x，保持值／dtype／对象／RNG身份；不计算、积累、缓存或拟合任何目标feature moments，不建style bank。算子及必要stateless moments stage没有参数、buffer或state_dict条目，不缓存特征图或带图张量。

推理仍`nativeROI RGB512 → 单完整encoder／GAP1／semantic1280 → 新point1280→1 → reading`。模型只接受image；内部源特征混合不混合标签、空间位置、ROI或GT，不代表channel统计只包含风格、不保证标签语义纯净或域不变性。

## 数据、优化和源选择

完全复用V22完成的22686 unique与V26的权威RGB512／output128／nativeROI／LINEAR／ImageNet normalize／原contrast,saturation .8–1.2与brightness±.1。旧18590每epoch一次＋4096roll四次=34974呈现／547批64末30；val原1506／24＋roll512／8=2018／32。四manifest rows／顺序／scalar标签／ROI不变；不加RGB、增强文件、源seed、渲染或GTcrop。128只是dataset接口，不生成geometry target；旧256合同仅population provenance。

原1506scene guard、14494／1506／4000与fixed ordinary1495／1497／severe6及focus保持。roll512独立layout／ID／group／declaredseed但sharedHDR，原20096metadata缺seed，不认证全历史seed链，不补评旧模型新roll或分辨率。

唯一scalar SmoothL1(beta=.05)。freshAdamW3e-4／wd1e-4／clip5／AMP default fresh GradScaler／seed20262020／train64val64workers4，全部CNN／point训练，BN仅源更新／eval冻结。encoder／point FP16、scalar FP32、style moments／混合 FP32后转回FP16；不扫描精度或scale。

源R=(原mean＋rollmean＋原tail10＋rolltail10)/4，normalized fullerror，失败1，tail最大ceil10%为151／52。earliest strict best、Plateau factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8及停止同R；min60max160，epoch≥60且epoch−best_epoch≥20停止，counter从1、LR／resume不重置，不凑旧68／109轮、不延长。完整model／optimizer／scheduler／scaler／RNG／history／totals／init／新protocolconfigcontract保存并严格恢复，拒旧方法；last只源。

## 必要新检查与执行

只新增CPU toy验证受控λ／π下公式、detached统计与内容梯度、eval精确identity且无RNG／统计调用、零参数／state、source-only单CNN／scalar、BN模式与严格新protocol恢复。真实model CPU构造0forward／0cachedweights／0RGB／0GPU确认metadata；旧GT／教师／G0／旧tests／数学或旧模型不重跑。

一次真实源AMP零更新见证恰3CNN／8图／1combinedback／0optimizer：4既有rolltrain0／1＋原val2＋rollval2。记录真实B×80×64×64 hook输入／FP32 moments与混合／FP16输出，style训练active1次4图、验证两次4图inactive，eval统计与随机调用0；完整cached462初始化逐项保持、semantic1280／point真实F16与scalarF32、encoderstem／neck／point梯度有限非零、233参数无optimizer不变、77BN源+1eval额外0、464strictreload、两val512identity。pref固定loss128再grad/128；正式fresh重建、defaultGradScaler不改、不沿用prefBN或随机状态，不额外GPU探测或扫seed。

V28整个source与I/R complete、ONE新鲜数字执行核对和规范化归档、旧父子退出／GPU实际空闲、本轮code/config冻结后，才启动唯一pref→source→Industrial→RF串行队列。V27失败资产保留，不恢复或把它作为完成的性能参照。活job只监控pipeline.pid／queue/currentstep原log／history/latest／实际父子GPU，不改livecode／模型／配置／名单／标签／loss／style算子／输入／BN，不加GPU或提前评估。

## 最终外评、成本与归档

source按冻结规则完整停止后，同一source-best对Industrial1395／52、RF100151／35各一次冻结评估，last不外评。两个已查看开发域不训练、总体feature统计拟合、BNadapt、checkpoint／epoch／早停／LR／训练超参／style层／α／概率／ROI／阈值／温度选择，不用目标病例设计标签。forward只有给定nativeROI RGB，RFscalar离线四点派生，input／scalar失败完整分母error1，不删行或把有效饱和值误判invalid。

E轮实际train34974E图／547E CNN及backattempt=updates＋AMPskip，val2018E／32E；target1546／25／0back／0update，总36992E＋1546图／579E＋25calls，point相同。style模块train547E／34974E图active，source eval32E／2018E图及target25／1546图只identity调用、active0／stats0／random0；实际counts分别记录。FP32 source moments混合的额外compute／memory不是免费，time／peak／init／stop／resume／R／LR与source-target隔离实记，pref单列，旧渲染不重复计费。

whole source＋I/R complete／四exit0／父子退出GPU空后才委派ONE新鲜独立CPU保存数字执行核对，实现作者不担任、不并行第二审计。收齐两源best/latest逐图、全部I/R scalar/group/summary、protocol/config/runmetadata/history/completed、实际CNN／point／style成本及执行证据；仅规范化explicit allowlist与taskcode推existing main，不branch／gitadd-A。

报告完整mean／Acc1/2/5／median/p90/p95／tail10／max、相对v16／旧GeoPRR／v24／v25／v26／v28及保存参照的>1/2/5新增救回、逐图组改善恶化、zero／nonzero／nearzero、ordinary1495／1497／severe6与固定focus。roll512只自身，V27无外域结果不拼造；源R或单域收益不代表整体成功，不自动promotion。所有旧正负资产保留，归档后按真人授权持续独立源域结构／表示迭代。

## Limitations

源特征的逐图均值／方差可能包含任务内容，混合可能损伤读数，不能声称只改变风格、标签语义严格保持或保证泛化。GAP／训练表示与历史相对V28同时变化，不作单因素因果；本轮固定变体不是原MixStyle完整配置的复现。单seed、已查看开发域、sharedHDR和历史seed缺失边界保持，不报SD、显著性、新盲测、普适性或新理论；旧GeoPRR历史pixel/fullweight链未重认证。RGB／weights／moments实际值／feature／geometry/H/C/XYZ/U/M／bbox／rawrun／fullcontracts／源arrays／privatepaths私有，仅公开规范化标量、形状／dtype和执行metadata。
