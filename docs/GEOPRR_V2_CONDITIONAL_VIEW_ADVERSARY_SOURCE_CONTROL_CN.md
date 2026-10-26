# 给定读数的source-view梯度控制协议

协议`geoprr_v2_conditional_view_adversary_source_control_v1`。新3代码与7项必要NEWCPU检查已完成；尚无source metadata执行、真实模型/权重读取、GPU或训练结果。这是有界source机制控制，不是正式V32、新理论或已推广模型。

## 问题与已有监督

原RGB identity15已展示source读数学习，丢弃原图信息的固定结构输入没有保留该能力。已有训练packet含2748 original和2748 roll、687完整parent及各view四state。它支持定义Z=0original/1roll这项生产程序metadata，不能由此假设h可识别Z，更不能把Z称实际camera角度、纯pose、domain因果或已证明的错误根因。旧absroll>90误差关联仍为否决；这里不使用其分组或病例。

问题是：给定已有scalar y后，可训练critic是否仍能识别source生产程序；若可学，反转这部分encoder梯度是否改善未见source布局读数并保护原准确图。保持原RGB输入、完整CNN/GAP1280/Linear1推理，不是再换pool/norm/读数head，也不引入q、可见mask、GT角色、teacher或scalar输出配对守恒。新增的是既有metadata辅助监督的训练消费，应明示不再是scalar-loss-only，未增加source像素或物理标签。

## 模型前metadata准入

第一stage仅CPU，最多30秒：既有5496行train manifest及prepared metadata，0RGB/header/factor/annotation/scene/model/weights/GPU。完整检查1374 viewquartet、687 original-roll parent闭包、2748配对state、标签0/1各2748且样本/顺序不变，无drop/resample。

标量必须直接沿用`experiments.geoprr_v2_data.normalized_target`后F32，与原Dataset实际y相同；只比较同parent-slot原/roll生产值差≤1e-6，该容差来自现有reader normalized-field一致规则。不是新GT公式、physical-y oracle或raw-label派生认证，不要求camera-roll前后物理角度triplet原值相等，也不捏造缺失normalized_progress/y字段。原reader近边界行为继承不改，不增加clip/校准。任何missing/invalid/pair不保持都在模型前整体`NOT_ADMITTED_SOURCE_VIEW_METADATA`，保存reason/partial，不改标签或子集救过。

生产值、Z、IDs、配对映射与完整合同仅serverprivate；stdout只counts/bool/maxpaired差等aggregate。Prepared metadata可能含完整nestedtrain/ID/OOD contract，parse/privatecopy不等于strict train-only整文件；此stage仅train行驱动y/Z，不打开ID/OOD manifest或图像。旧q partial/cache不读取。

## 原RGB路径与训练critic

主模型fresh existing-cached完整ImageNet MNV4 num_classes0、conv_head/norm_head/GAP1、语义1280、新Linear1280→1 std.001/bias0，原encoder462 states保留。NativeROI512/LINEAR/photo/sourceBN/encoder-pointAMP/scalarF32不改。

训练helper只用临时local point prehook捕获h[B,1280]，原reader.forward不改，finally移除且不跨forward缓存/输出features。训练critic为F32 Linear1281→64/ReLU/Linear64→1，含bias、无BN/dropout；输入concat(GRL(h.float),stopgrad(y.float))。Critic构造在main之后的forkedCPU RNG context内，恢复globalRNG、不加实验seed，两arm同初值。

额外critic82113参数/4tensors；完整预期8517906参数、237可训练tensors、468states（encoder462+point2+critic4）、原77BN。以上必须后续检查/真实pref确认，当前未观察。Eval直接原image→scalar，不执行critic、不接受y/Z/partner/template/GT或拟合校正。

GRL是无参数/无state的identity forward，只对h backward乘-alpha，detached alpha0、reversal alpha.05；critic参数仍优化未缩放meanBCE，source y conditioning不回传。Forward数值为原meanSmoothL1beta.05加meanBCE，记录两primitive与forward sum；反转encoder并非在最小化这个共同sum，不能误报优化目标。Critic/concat/logits/BCE在disabled-autocast F32，主CNN/point原AMP。

## 两arm顺序和固定预算

1. Metadata准入全通过后，一次必要source0opt pref，正式arm都fresh，不继承pref状态。
2. Fresh detached-control15E/last15，前14轮无validation、无best/scheduler/延长。与原source recipe相同：AdamW3e-4/wd1e-4/clip5/defaultfreshGradScaler/seed20262020/batch64末56/workers4/loaderseed+epoch、5496每E一次、原contrast/saturation0.8..1.2 brightness±0.1，无新空间或输入变换。Jointclip包含critic，两arm一致；不声称control权重轨迹等同旧identity。
3. 末native292一次。须先通过constant0.5 mean和ceil30tail各半的原学习gate，且末轮在线critic正确总数/5496≥.65，否则`NOT_RETAINED`，CF/reversal均null/0calls。该失败只是固定预算下未展示必要可学习性，不证明Z无信息或所有GRL无效。
4. 以上两项均过才完整评分control sourceCF292 rawRGswap[1,0,2]一次，然后fresh同15E reversal。不能按controlCF调参或改checkpoint；唯一机制干预是encoder critic梯度0→-.05。
5. 末reversal native292一次，原constant学习gate和identity/control native联合保护全过才同last15评分CF292一次。Critic effect不提前阻断该CF，完整分开报告source读数与有限critic效应。

Zhat固定logit≥0。Critic accuracy是末轮全部5496在线呈现的correct_count/total，包含AMPskip批，labelcounts各2748；不是batch均值、冻结checkpoint额外forward或held-out probe。正常AMPskip保留，backattempt=updates+skip，不调scale或补更新。

## 联合接受

Identity15 native mean/tail/max为0.025431218243695233/0.11800955331418664/0.5379703640937805，>5=16，≤1%准确102；CF为0.026353762864682596/0.12945961876345488/0.5424728989601135，>5=17，其自身准确104。

Reversal native须mean≤reference、tail严格低、max≤、>5≤16、native准确102无新>5；CF须mean/tail/max≤reference、>5≤17、其自己准确104无新>5。另在native/CF各自相对新control须mean≤、tail严格低、max≤、>5≤，不得把该view control自身≤1%图变>5。四个准确集独立推导/保留IDs，不拿native替CF或跨checkpoint拼最佳。

再要求有限训练critic effect：末轮reversal在线accuracy在[.4,.6]，且abs(controlacc-.5)-abs(reversalacc-.5)≥.10。所有条件在新模型前固定；读数过但effect不过，报告mixed/notretained，不自动保留或promotion。Co-trained准确率受容量与动态影响，不证明I(h;Z|Y)=0或真实DG。

仅全部native/CF readingjoint与critic effect通过才`CONDITIONAL_VIEW_SOURCE_CONTROL_PASSED`；否则`NOT_RETAINED`或更早metadata拒绝。完整分母292，input/scalar失败error1，0/1合法饱和值非failure，不删行。报告mean/Acc1/2/5/median/p90/p95/tail/max、逐图组改善恶化、三阈值救回新增、准确集、criticBCE/correct/total/classcounts与所有cost。未运行是null/0，不能将历史referenceCF当新模型结果。旧drift阈值只诊断。

## 计数、检查与保存边界

每arm82440 train图/1290 CNN-point-critic calls各，1290 combined CNN-point-critic backattempt=updates+skip。Control前提失败总82732 CNN-point图/1295calls，critic82440/1290；controlCF完成83024/1300。两arm但GRLnative失败总165756/2595，双CF最大166048/2600，critic164880/2580、2580backs。Metadata0CNN/RGB，pref另列；不把计划预算算作未执行事实，不把critic/metadata成本算免费。

必要NEWCPU只新GRL sign/critic梯度/y-stopgrad/F32autocast/forkRNG/localhook/eval0、metadata-Z和mock生产值consumer、阶段计数/strict恢复/partial。0realMNV4/forward/cachedweights/RGB/GPU，不重跑旧q/variance/GT/G0/selector/tests。仅新生成的isolated synthetic toy-state可序列化reload，不是真实checkpoint或权重。

Pref只一次、alpha.05：既有fixedpref train4 original/roll slot0（canonical01/02、Z2/2），再ID两batch各2，合计8unique/3CNN-point、critic4example/1call、1combinedback0opt。另同4ID CPU identity再decode，共12decode而非8；ID/CF不提供y给critic。Loss128后grad/128，正式freshdefaultScaler不携pref。见证cached462初值/237params0opt不变/468reload、77BN+1eval0、h1280F16/pointF16/scalarF32/critic1281与BCEF32、stemneckpoint和critic首尾W梯度存在finite非零（bias可抵消0）。精确GRL符号由CPUtoy验证，不加真实forward/back。

入口计时在torchimport前，metadata/各arm/train/native/CF/fullwall/峰内存/critic/hook/BCE实测或unknownnull，point独立未知为null。Finalize子模块用时不是wholepipeline，父supervisor/screen/queue边界单列。Strict新protocol/config/arm/alpha/source-metadatacontract/fullreader+critic/optimizer/scaler/RNG/history/totals completeepoch恢复，不兼容旧464/465或他版本；FileExists/partial/无summary CSV保护不绕、不二forward。

代码/config/协议/必要NEWCPU existingmain冻结scoped部署，actualGPUidle后唯一串行。Wholecomplete且所有实际父子gone/GPUempty后，ONEfresh非作者stdlib saved-number/execution CPU核对再explicitallowlist归档，不平行二审/重新RGB或weights/GT数学。元数据拒绝时没有训练成绩，仅核已保存reason/counts。

## Limitations

Source训练使用提供的y和既有viewmetadata，而非新增物理GT认证；原reader近边界规则不改，C3原149/148/1 rawderivation资格不清PASS。一个source procedure不等纯camera/真实domain或错误原因。Onlinecritic不是独立probe，不能证条件独立、因果、新blindtest、显著性、收敛或真实泛化。旧q/Sobel/G1/teacher/anchor/VDN/virtualQ/配对守恒等关闭决定保持。

Source03与真实Industrial/RF不读像素/模型/训练、adapt、statfit、BNadapt或训练/ROI/temperature/threshold选择；不新source数据/RGB/render/资产/物理标签/weights/foundation/seed/买算力，不改系统PythonCUDAconda其他项目/清旧资产/officialtest4000/第三方消息/新branch。Rawcontracts/训练IDs/y/Z/特征/权重/RGB/privatepaths留server，stdout先allowlist仅aggregate/bool/允许scalarCSV。未来正式模型必须另立完整source预算/R/stop，source-best对两个真实域各一次后再唯一核对；本控制正结果不自动正式长训或promotion。
