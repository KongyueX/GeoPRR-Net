# 可逆颜色通道变化：有界source-only控制

本控制检验一个明确的输入问题：保留原模型与scalar标签，只在源训练中循环RGB通道身份，能否保持原生源读数，并使同一模型对训练子群外的一次固定通道交换保持读数稳定？这是内部有界source控制，不是正式V32、完整泛化模型或已证明的失败机制。

## 源证据与方法范围

现有可追溯源共享HDR；同layout四state主要改变r，并没有提供固定g/r/c、仅a变化的已确认像素组。历史parent精确重放未达到原实现provenance门槛，已经停止，不借本控制换parent／seed／renderer重试。当前可见针体／刻度监督路线也已因材料／像素语义未建立而否决；不把projected vertices伪称visible mask，不增加几何GT。

[Qi等，DoubleAUG，arXiv2023，§3.1](https://arxiv.org/html/2311.13198v1)提出在single-domain object detection中交换RGB通道。这里仅采用一个固定C3子群和预留transposition，不使用其Dual-Style Memory，不复现整篇方法，也不据作者分数推断仪表收益。[RandConv，ICLR2021](https://openreview.net/forum?id=BVSM0x3EDK6)仅提供输入外观随机化的一般背景，本控制不运行随机卷积。

通道置换保留完整三通道数组的信息、空间坐标及物理scalar标签，不等同于新物理光照、domain独立性、可读性认证或新RGB资产。已有源state响应并未证明color cue是错误原因；本控制可以失败，不追加颜色／模型／seed／预算扫描。

## 固定模型与输入

完全保留fresh cached ImageNet MobileNetV4 conv-medium、完整conv_head/norm_head/GAP1/实际semantic1280、新Linear1280→1（std .001/bias0）、source BN、原AMP与FP32 sigmoid。模型8435793参数、233参数tensor、464states、77BN。无新head/pool/norm、teacher、adapter、geometry、pair loss、feature-stat fit或新weights。

复用既有server冻结source packet：5496训练图、687parents（两seen针型各原／roll两view）、source-ID292图／73layouts。相同512 nativeROI、LINEAR、ImageNet归一化及原train photo不变，128仍只API。旧源1506／14494／4000和保护名单不动。prepared contract可能校验已有源OOD元数据，但源needle03完全不读RGB、不forward、不计分；Industrial1395／RF151全部0调用，永不参与训练／适配／BN／checkpoint或超参选择。

训练order固定为RGB/GBR/BRG三个C3元素，编号为`(dataset_index+epoch-1)%3`。每轮5496图各一次、三order各1832；固定15轮，每张raw源图每order恰5次。算子在原photo之后、encoder之前，对normalized FP32 RGB实施：

`zprime[c] = std[perm[c]]/std[c] * z[perm[c]] + (mean[perm[c]]-mean[c])/std[c]`。

identity直接保留原tensor。无额外RNG、clip、grayscale、resize、物理标签重算、RGB写盘、feature cache或可学习参数。它等价于raw RGB换通道后使用同一ImageNet归一化。

## 训练、门槛与停止

固定fresh AdamW lr3e-4、wd1e-4、clip5、SmoothL1 beta .05、default fresh GradScaler，seed20262020、batch64、workers4、原loader seed20262020+epoch。固定15轮、无scheduler、前14轮无验证、无best选择或延长，仅last15。与旧directFT5相比，raw-image与CNN更新预算明确为3倍；这是每个C3 order各5次呈现的资源cap，不是同预算因果比较，也不复训旧baseline。

last15先只评完整原生source-ID292一次。保留失败分母error1，0/1有效饱和值不误判为失败。先过原constant0.5双学习gate（mean≤12.504463703047%FS、ceil30 tail≤23.692208052865%FS），并同时满足已保存directFT5参照：mean≤6.132800787730%、tail<25.669996334861%、max≤57.870632410049%、>1/2/5计数≤245/209/114。全部满足才进入counterfactual步骤；否则正常停止，后一步0CNN且无结果文件。

counterfactual仅同一last15、相同292源ID／label／ROI，用预先固定的raw-RG交换`[1,0,2]`各一次。它在训练C3之外，但仍是同layout人工counterfactual，不能称新独立域／盲测。须同时满足mean≤12.504463703047%、tail≤23.692208052865%、相对native的平均绝对prediction drift≤1%FS、ceil30 drift≤5%FS。Invalid行的reading error和drift均记1，不删行。完整mean/tail/max、Acc1/2/5、median/p90/p95、fail、逐图新错／救回及drift均保存。

只有全部门槛通过才记`SOURCE_COLOR_CONTROL_PASSED`；否则`NOT_RETAINED`。通过只供root考虑下一独立正式source实验，不自动promotion、外域评分或长训。没有任何门槛可按观察结果后改。

## 必要检查、执行与成本

只新增normalized-channel公式／inverse／identity／无RNG、C3 exposure及新增native-primary／CF条件和drift／identity／计数／resume元数据toys。0真实MNV4构造／RGB／weights／GPU，不重复旧selector、constant gate、GT、教师、数学或旧tests。代码／协议／config／新CPU证据先冻结push existingmain，严格scoped部署新3taskcode和所需3metadata，确认实际旧父子gone及GPUidle后才唯一pref→source控制。

一次新源pref为3CNN／8不同RGB／1combined backward／0optimizer：4旧固定seen train采用order0/1/2/0，source-ID每seen针型字典序前2图，共4图分两eval batch。核实cached462初值、233参数0opt不变、464strictreload、77BN train+1/eval+0、实际operator FP32与semantic/point F16、scalar F32、stem/neck/point梯度有限非零及native input identity。RG算子仅在已读training tensors上做0额外CNN的inverse检查；不提前跑held-out view模型。正式fresh重建，无pref BN／RNG状态继承；formal默认Scaler不调。

| 正式步骤 | CNN/point图像呈现 | CNN/point调用 | backward尝试 |
|---|---:|---:|---:|
| 15轮训练 | 82440 | 1290 | 1290 |
| native-ID | 292 | 5 | 0 |
| 条件CF-ID | 292 | 5 | 0 |
| native未过总计 | 82732 | 1295 | 1290 |
| CF完成总计 | 83024 | 1300 | 1290 |

pref另列。每次backattempt如实分解updates+AMPskip；不调scale补齐。完整entry timer必须早于torch import，phase/wall/peak按真实边界保存，point独立时间未知写null。新protocol/config/data与全model/optimizer/scaler/RNG/history/totals严格保存；无latest不能fresh覆盖，有模型partial不复用zero-stage恢复；已存CSV无完整summary不得再次forward。

控制wholecomplete且父子gone/GPUempty后，仅ONEfresh独立保存数字CPU核对，作者不能审计、不并行第二审计、不打开weights。规范化explicitallowlist归档existingmain，保留失败／混合结果与全部成本。RGB、weights、features、fullcontracts、训练身份数组和geometry保持server私有，stdout前server allowlist，只收必要metadata/bool及允许的scalar CSV。

## Limitations

这是固定15轮source控制；额外预算、C3子群和原source-ID重复使用均明确披露，不能把native改善归因于颜色算子或称新盲测。人工channel swap并不覆盖材质、光照、真实sensor或不同表结构；source门槛通过不保证真实外域提升。G1／anchor／teacher／65轮／配对守恒及全部旧结果保留。旧GeoPRR历史pixel／fullweight链仍未重新认证。无新seed、foundation、weights、算力购买、officialtest4000或第三方消息。
