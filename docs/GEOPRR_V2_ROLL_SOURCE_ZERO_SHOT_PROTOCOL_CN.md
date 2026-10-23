# 独立真实相机roll源覆盖：zero-shot实验协议

2026-10-24。v22冻结一个性能／泛化优先的独立源域实验：完整ImageNet MobileNetV4直接标量读数，加入真实相机局部光轴连续roll训练图，以及独立布局的源roll验证。预算由预先声明的源学习规则决定，不要求与旧实验相同。本文给出数据、训练与评估合同，不把尚未生成／运行的工件或预算写成性能结果。Industrial1395与RF100151严格只在训练结束后对冻结best评估；全部旧模型、原G1／anchor失败、65轮与负结果保留。

## 问题与唯一模型

现有源资产盘点显示，renderer已经随机化字体、字号、刻度、颜色和三种现有针型，原数据也有部分非零roll；不能宣称这些因素完全缺失。更具体的缺口是相机roll没有作为独立于物理状态的明确源因子，表盘基础布局、绝对针方向和读数仍有关联。现有本地可重渲染HDR只有1个，原源数据的131种背景名称不等于本地拥有131套HDR，不承诺新增背景域。

本轮保留v16的单图部署表示：fresh已有缓存的`mobilenetv4_conv_medium.e500_r224_in1k`，`num_classes=0`，完整conv_head／norm_head及1280维semantic接Linear1280→1与sigmoid。预期8435793参数／464 state，由本轮实际构造记录核实。全部encoder和point训练，source BN正常更新，eval固定。输入为单张ROI RGB，OpenCV LINEAR到256、ImageNet归一化，没有orbit pooling、额外几何读出、router、ensemble、TTA或新loss；目标每图仍只走一次CNN。

不加载任何训练好的任务checkpoint，不延长v16／v21，不下载新foundation；HF_HUB_OFFLINE=1，仅使用已有ImageNet缓存。推理只接收RGB，source H、field、center、SPD、角色或针尖不进入模型；唯一训练监督是物理归一化scalar y，loss为SmoothL1(beta=.05)。本轮改变源覆盖、重复曝光、源验证选择与训练停止规则，不能将差异单因素归因于相机roll。

## 新真实源图与物理合同

保留当前18590 unique训练图及其全部标签／路径，含v11已有端点替换。另从原1024个训练布局生成4096张新RGB，每组4个状态，采用当前源manifest对应状态：首128布局为[0,.005,.02,1]，其余896为原quartet状态。新roll图是追加数据，不替换或覆盖旧RGB；原14494、原1506验证与4000 officialtest保持。

另用128个新布局生成512张源roll验证图。布局由现有`generation_plan(groups=128,seed=20270000)`的原quartet规则产生；generated render seeds为20270000..20270127，与已声明1024训练布局的20262020..20263043不重叠。这里声明的是新布局／ID／group／generated seed独立，不冒称原14494历史seed全链已重新核对。新验证与新训练共享现有真实source-train HDR；它不是新背景、scene holdout或新域盲测。原1506仍保留原scene holdout。

roll预先由独立`np.random.default_rng(20261024)`顺序生成全部1152个`uniform[-180,180)`角度，1024训练组与128验证组各取计划中的一个角度。同组四状态共享roll，roll RNG与状态采样分开；不依据两套外域或源失败案例挑角度，不因生成失败重采样。新验证quartet不保证含精确0／1，须如实报告其端点人数；不能把源roll验证写成已覆盖全部端点的保证。0／1指归一化量程起点／终点，0不必是绝对物理读数零。

在原`move_camera`完成后、首次RGB渲染和标注投影导出之前，对原camera quaternion右乘局部z轴roll quaternion。倾斜相机不得用全局Euler z替代局部光轴变换，也不能只旋转旧RGB并复用旧像素几何。实际相机姿态、H、bbox与投影必须由新渲染重新导出。针的物理旋转、刻度值、跨度、网格、颜色和世界光照等非相机因素沿用相应布局；同组内除状态外保持固定。相比旧组，允许的camera变化及其导出投影变化是新合同的一部分，必须同时反映snapshot的g.objects Camera与c，不能沿用v11“旧新g/a/c精确相等”的替换声明。

正式渲染仍为1920×1080、JPEG95、Cycles CUDA、200 samples、4 threads，全部4608张真实新RGB。针型仍是现有三种固定网格，不同时增加配重形变、字体、背景或新模块。计划、parent layout／sample身份、split、state policy、vertex order、render seed、roll及姿态／来源记录保存在私有工件中。

新生成须逐组检查实际投影／bbox可见性、physical y与有向刻度／跨度、渲染姿态与metadata一致及同组非状态固定性。新prepare复用`prepare_compositional.prepare_group`，对这4608张新图按默认768／192完整`geometry_targets`构造Gaussian Q、M和U，每组执行16次Q×M×U cross-state组合GT读数检查，保留min_support_mass=.15、max_readout_error=.01（1%FS）及原物理限制；全部1152组对应18432次组合检查。该检查只针对本轮新生成数据，不是重审旧18590，也不代表256／64网格的Q检查通过。新训练仍只消费256 RGB／y，不把这些GT几何输入模型或增加loss。失败保留partial并停止，不删失败组、不裁剪GT、不重采样角度或放宽保护，旧数据及其保护保持。

## 两种源验证边界

新源根为`rendering/generalization_roll_v22_20261024`，私有manifest约定为`manifests/train_roll.jsonl`（4096 unique）、`train_combined.jsonl`（22686 unique）与`validation_roll.jsonl`（512）。必须保留真实source scene字段，不伪造scene名称通过原split检查。

全部22686 unique训练行与原1506验证仍完整执行原`source_split_check`及既有scene／人口保护。新512验证独立使用明确的新布局合同：与训练无相同layout／sample ID／group／已声明generated seed，同时允许同一HDR／source scene。不能把它塞进旧scene-holdout合同后绕过失败，也不能将这一新合同描述成原scene holdout已经通过。实际实现须分别记录旧边界和新布局边界，且两套验证不参与梯度、BN或训练总体统计。

## 重复曝光与源学习规则

每epoch构造精确索引multiset：旧18590各一次，新4096各重复4次，再按源seed与epoch洗牌，`drop_last=False`。不是按概率抽样使次数仅在期望上相符。每轮34974次RGB呈现／547个batch64，末batch30，unique仍为22686；16384次新roll呈现占46.8462%。重复使用同一固定物理RGB，每次独立采样原contrast／saturation 0.8–1.2和brightness ±.1，不能称为四次不同roll或四份独立渲染。

首128新roll组含128张归一化起点图，经4次曝光为512次，加旧源128次，合计640／34974≈1.83% exact-start训练曝光；旧v16为128／18590≈0.6885%。这也是数据分布改变的一部分。不要把重复采样增加的曝光混同独立布局、针形或HDR多样性。

fresh AdamW，初始LR3e-4、weight_decay1e-4、gradient_clip5、AMP、seed20262020、train64／val64／workers4，最少60、最多160完整epoch。唯一源选择分数为：

`score = 0.5 * (mean_normalized_error(original1506) + mean_normalized_error(roll512))`。

每个cohort各自保留完整分母，输入或scalar失败按normalized error1计入。两cohort等权是预先固定的学习目标，不能改成2018行合并均值；报告必须同时给出两个cohort的原始误差、人数及score。只按未舍入score的最早严格最小值选best，minimum delta=0。

每轮完成两套验证后，对同一score调用一次PyTorch ReduceLROnPlateau：mode=min、factor=.3、patience=8、threshold=0／abs、cooldown=0、min_lr1e-6、eps1e-8；遵循该scheduler容忍8次未改善、之后未改善才触发的实际语义，降低后的LR供下一轮使用。记录实际LR与触发，不能将scheduler当作checkpoint选择或改用另一套指标。

早停counter从epoch1开始，只有严格新best才重置；scheduler降LR不重置早停counter。完成一轮后，若`epoch>=60 && epoch-best_epoch>=20`则停止，因此最早可以在60轮结束；否则继续至160轮兜底。不是从60轮才开始累计20轮。latest必须保存scheduler、best score／epoch、counter、optimizer／scaler／RNG与累计成本，恢复不能重置plateau或额外延长上限。last为实际停止轮次，不预设为60或160，仅保存源结果。

## 冻结目标评估与实际预算

只有源训练按上述规则结束并锁定best后，才各一次冻结Industrial1395／52组与RF100151／35组评估；last不追加外评。目标严格不用于训练loss、总体特征统计拟合、BN适配、source score、checkpoint／epoch、LR、停止条件、阈值、ROI、温度或其他训练超参数选择。推理只有原生ROI RGB／256 LINEAR／ImageNet／eval，没有GT几何或目标条件路由。RF100 scalar仍由四点离线派生，全部目标输入／scalar失败保留error1。

若实际训练E轮：train为34974E图／547E次CNN及反传尝试，validation为2018E图／32E次CNN（原1506／24与roll512／8分开）；两个外域1546图／25次CNN、0反传／更新。60轮下界计划总2221066图／34765次CNN，160轮上界5920266图／92665次CNN；实际有效updates＋AMP skips必须等于547E。生成4608张RGB的渲染／prepare成本、物理数据核对、运行见证及训练／验证／外评、process／full时间、峰值内存、resume和存储分别实记，不报计划为已完成。

本轮不要求旧预算对齐，也不新增预算匹配对照来阻断性能实验。与v16／v18／v21及旧GeoPRR比较时，直接报告更大的源图／曝光和不同验证／优化规则；同架构差异不构成单camera因果证明。只复用旧模型保存参照，不自动重跑旧模型、官方test、seed或追加target调参。

## 入口与完成

源生成controller协议为`geoprr_v2_camera_roll_source_v22`；训练协议为`geoprr_v2_roll_source_full_semantic_direct_source_only_v1`，入口`experiments.run_geoprr_v2_roll_source`，输出`runs/geoprr_generalization/iteration_v22_roll_source_20261024/source_only_training`。两阶段独立记录实际执行版本与snapshot，当前[配置](data/geoprr_v2_roll_source_20261024/config.json)与训练器CONFIG对齐，生成provenance另记，不能把生成字段混入runtime CONFIG。 实现入口见[真实roll渲染](../experiments/render_geoprr_v2_roll_group.py)、[新源物理准备](../experiments/prepare_geoprr_v2_roll_source.py)、[标量训练器](../experiments/geoprr_v2_roll_source.py)与[唯一supervisor](../experiments/run_geoprr_v2_roll_source.py)。

先完成全部新源生成与物理准备，保留原资产；源生成阶段CPU准备时GPU暂时为空不意味着可并起训练。确认source complete、源生成controller／子进程退出且GPU实际空闲，由唯一supervisor继续阶段交接，配置／代码／协议冻结后，进行一次必要的新RGB／loader／模型运行见证：必须消费实际新roll源图，核实精确重复索引、两套验证分母／score、256预处理、单图纯RGB完整semantic／point、有限loss与encoder／neck／point梯度、0optimizer、source BN更新／eval固定、完整实际state严格reload。训练器当前计划见证3次CNN／8图／1次反传／0更新，实际counts与状态以执行记录为准，不能提前称已通过；正式训练重新fresh初始化，不复用见证BN。

唯一GPU队列串行source生成→新见证→源训练至停止→Industrial→RF100。运行读pipeline PID、queue current_step、原日志、history／latest与实际GPU命令，不以best更新时间判断停滞，不重复阶段交接中的活job。活job期间只监控，不改model／code／config／人口／标签，也不额外GPU评估。

真实实现故障先读当前失败step原日志／活命令，保留partial、failure、原protocol与snapshot；必要真实修复后在相同output从latest或未完成的源组恢复，不能按旧PID重跑、删样本或绕保护。实质方法／生成配方／名单变化另立独立实验。只用项目环境、不买算力或新模型、不删旧资产；源盘点估计新RGB与factor约1.89GB，实际容量与成本仍按生成记录，空间不足先保留partial，不能自动清理旧实验。

整个新实验完整结束后，收齐两套源best／last预测、source score／stopping／scheduler证据、冻结外域完整逐图／组／summary和真实成本，仅一次必要独立CPU数值／执行核对后形成完整结论。报告各源／外域NMAE／Acc、zero／nonzero／nearzero、组与roll源覆盖、原ordinary1495／1497／severe6、重点3172／2810／3663／9323／8195、sync9／4025／338／4658及主要目标大错、tail／max／救回新增，不根据单个旧门槛自动否决整个论文或删除负结果。只有完整结果再通知，不以启动／中途快照作周期性提醒。

仅本任务代码及显式规范化allowlist可推现有main，不建branch；RGB、权重、H／C／XYZ／field／bbox、特征矩阵、完整源合同／计划逐行数组与private trace保留私有。

## Limitations

新roll验证只有布局／generated seed／ID／group独立，背景与现有源训练共享；它是用于选择的源验证，不是外部或背景盲测，原1506的旧scene holdout另报。当前单个HDR与三种现有针型没有变成新背景或连续针形覆盖。新增数据、46.8462%重复roll曝光、端点曝光和源验证／adaptive stopping一起改变，不能单因素归因或声称严格预算公平。Industrial／RF100已被查看、单seed不构成新盲测、SD、显著性或普适机制；旧GeoPRR历史执行像素／完整权重链尚未重新认证。源score改善也可能不改善目标；保留全部连续收益与尾部代价，不预先承诺成功或自动升最终模型。
