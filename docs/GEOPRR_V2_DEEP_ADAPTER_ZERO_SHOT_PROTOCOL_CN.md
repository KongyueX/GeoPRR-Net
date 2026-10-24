# 固定完整预训练表示的深层适配：zero-shot协议

2026-10-24。V24是独立fresh源域结构／表示实验。固定完整缓存ImageNet MobileNetV4原始参数与所有BN状态，在两个深层空间表示内部插入可训练残差adapter，以完整1280维语义形成唯一标量。数据、光度、scalar loss及源mean-tail风险选择沿用V23；不继续堆latent queries，不追加源图或续训旧模型。实际CPU零前向构造已确认8910793参数、476281可训练参数、241个参数tensor（10个可训练）与472个state；真实RGB执行仍待本轮见证，本文不预报性能。

## 源证据与可证伪问题

V23源best97的原1506 mean／tail10／max为0.490210／2.440468／62.204719%FS，roll512 mean／tail10为0.740396／3.029107%FS。低平均误差仍可伴随极大错。V23全部原CNN参数与BN更新，完整semantic1280先压成128维query上下文；这些是实现事实，不足以证明严重错误由遗忘造成。

本轮检验固定原预训练权重和统计、只在深层空间表示内部学习任务变换时，能否充分拟合源任务，同时改善完整源mean-tail风险与最终外域泛化。原始[残差adapter论文](https://arxiv.org/abs/1705.08045)仅提供共享视觉网络内部适配的一般动机；当前两层瓶颈、固定BN与标量任务不是论文复现或泛化保证。结构与训练参数范围、BN政策同时改变，不能单因素归因于避免遗忘。

## 唯一RGB部署通路

fresh完整`mobilenetv4_conv_medium.e500_r224_in1k`，`num_classes=0`，只从已有缓存加载ImageNet权重，`HF_HUB_OFFLINE=1`。原8434512参数／462个state tensor、conv_head／norm_head及实际semantic1280保留；原参数含BN affine全部固定，所有原BN running stats和77个counter在源训练及eval均固定。原`num_features=960`不是semantic输出。

reader外层`ModuleDict`持有两个adapter，分别通过forward hook替换`encoder.blocks[2]`和`encoder.blocks[4]`输出，原encoder的state key保持原名。原输出分别为160×16×16和960×8×8，对应瓶颈40／240。每个adapter为：

`x + Conv1x1(C/4,C,bias=True)(GELU(Conv1x1(C,C/4,bias=True)(x)))`。

down使用Conv默认权重初始化、bias为0，up的weight／bias均为0，因此初始残差恒等。先按V16约定新建point1280→1（weight std=.001、bias0），再构造adapter；不加载旧task point或checkpoint。完整encoder的1280语义经point，在logit转float后sigmoid，唯一输出B标量。

`RGB256 → 固定原encoder＋两个内部可训练adapter → 完整semantic1280 → 新point → scalar`。

`train(mode)`保持encoder.eval()，只让adapter和point进入相应模式。不得以`torch.no_grad()`包围整个encoder：第一个adapter的梯度须穿过固定的后续CNN和neck；原参数无需梯度。两个1×1 adapter重新组合既有空间位置的通道，不新增空间采样或恢复早期细节。

没有query／FPN／旧模型残差或输出融合／router／teacher／GT角色／ring／几何读出／CDF／置信gate／TTA。模型只接收image；GT几何和目标scalar不进入forward。实际CPU零前向构造确认总8910793参数、476281可训练参数、241个参数tensor（10个可训练）与472个state。完整原encoder和全部新模块均保存，不能仅存adapter而让恢复静默更换缓存。

## 数据、源监督及停止

完全复用已完成V22源：18590旧源＋4096真实roll＝22686unique；每epoch旧源一次、roll四次，34974呈现／547个batch64末30。源验证原1506／24次调用和roll512／8次调用独立报告。256 OpenCV LINEAR／ImageNet及contrast／saturation .8–1.2、brightness ±.1保持；不重渲染、改名单、增强、标签或旧GT。原14494／1506／4000及全部固定cohorts／focus保留。

roll512只在layout／ID／group／声明生成seed层面独立，共享训练HDR；原20096 metadata缺历史seed，原1506 scene guard保持，不认证全部seed链或新域盲测。

唯一源loss为scalar SmoothL1(beta=.05)。optimizer仅接收两个adapter和新point的10个参数tensor，clip也仅作用于这些trainable参数。fresh AdamW LR3e-4、wd1e-4、clip5、default fresh GradScaler／AMP、seed20262020、train64／val64／workers4，不扫描LR／adapter宽度或精度。

源风险固定：`R=(mean_original+mean_roll+tail10_original+tail10_roll)/4`。四项normalized full-denominator error，tail严格最大ceil(.1*N)，分别151／52行，失败error1进入mean／tail。最早严格最小R选best，同一R驱动ReduceLROnPlateau（factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8）和早停。

至少60、最多160完整epoch，`epoch>=60 && epoch-best_epoch>=20`停止；counter从1起，LR下降和resume不重置，不延长上限。last只两源。完整保存model／optimizer／scheduler／scaler／RNG／history／totals及初始化，严格本轮protocol／config／source合同恢复。每epoch直接比较原462个state与设备上的fresh缓存基准，确认原权重／BN全部不变；不新增hash或反复CPU下载。

## 必要新检查

只检查新内部适配和冻结语义，不重跑V23／旧pref、GT、教师、G0、旧tests或数学审计。CPU toy核实零up恒等、非零up实际改变下游输出、第一up梯度穿过固定后续网络／neck、初始down梯度存在且为0、独立toy非零up后down梯度非零、B1／B3与partial batch、train／eval／train固定全部原state和BN、RGB-only、strict state及源R／停止接口。真实模型CPU零forward构造确认实际参数／state，不将尺寸推算说成实测。

一次真实源零更新见证：4张既有roll训练0／1图＋两源验证各2张，3次CNN／8图／1次combined backward／0optimizer。前向autocast F16；pref反传固定loss×128后grad÷128，正式default GradScaler不变，无scale扫描。记录两adapter实际形状与调用、semantic1280、up和point梯度finite非零、down存在且为0、原参数grad为None、全部462原state和新参数无optimizer不变、77BN计数器及running stats增量0、两验证256输入identity和完整state严格reload。

初始zero-up导致down首步零梯度属于预定链式法则，不能误报故障、篡改up初始化或要求所有新参数首步非零。正式训练fresh重建，不沿用pref参数／状态。没有另一个旧模型或目标GPU对照。

## 外评、成本与归档

V23已完整完成并经唯一CPU核对与规范化归档后，本轮才串行运行pref→source→Industrial→RF100。完整源停止后，冻结sourcebest各一次Industrial1395／52和RF100151／35；last不外评。两域不参与训练、总体特征拟合、BN适配、checkpoint／epoch／早停／LR、ROI／阈值／温度或训练超参选择。原生ROI RGB/eval，RFscalar离线四点派生，输入／scalar失败完整分母error1，不删行。

E轮实记train34974E图／547E次CNN和loss backward尝试，val2018E图／32E调用，updates＋AMPskip=547E；外域1546图／25调用／0back／0update，正式总36992E＋1546图／579E＋25调用。两个adapter调用每次与CNN一致；原参数冻结不意味着固定后续网络输入梯度反传免费，也不称整个原encoder参数训练。pref成本单列，旧源生成不作为本轮新成本。另记时间、peak、stop、R、初始化、冻结检查与resume。

完整结束后ONE独立CPU保存数字／执行核对，不由实现作者再作第二审计。报告两源best／last、外域mean／Acc1/2/5／median/p90/p95／tail10／max、>1/2/5救回新增、zero/nonzero/nearzero／groups、ordinary1495/1497/severe6及固定focus；不重评旧模型或对roll512补旧分数。只公开规范化explicit allowlist及本任务code，原RGB／weights／H/C/XYZ/U/M／bbox／features／rawrun／fullcontracts／sourcearrays／privatepaths保持私有，仅推现有main。

## Limitations

冻结原权重／统计不保证训练后适配表示或泛化语义保持；深层pointwise低秩变换可能欠拟合，不能新学空间卷积或恢复输入缩小丢失的信息，适配输出与固定BN尺度也可能失配。没有语义遗忘因果证明或实体定位认证。单seed、两外域已查看为开发人口，旧GeoPRR历史pixel／完整权重链未重新认证，不报新盲测、SD、显著性或新理论。均值、尾部、最大错与错误交换共同报告，未达标保留完整负结果，不自动升级模型。真人已授权持续独立源域结构迭代，单轮按冻结规则结束后继续研究，不延长已结束实验或重复外评。
