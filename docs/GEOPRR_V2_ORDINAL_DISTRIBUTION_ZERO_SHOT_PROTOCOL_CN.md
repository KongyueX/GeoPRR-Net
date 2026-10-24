# 直接有序读数分布与CDF log-score：zero-shot协议

2026-10-24。V25为新的源域表示／监督实验：完整fresh缓存ImageNet MobileNetV4端到端学习，完整semantic1280接唯一256节点读数分布头，使用唯一有序CDF Bernoulli log-score，部署固定全质量连续期望。已有数据、光度、源R和停止规则保持，不复用旧task权重、adapter、query或几何读出。实际CPU零前向构造已确认8762448参数／233个参数tensor／465 state；真实RGB见证仍待执行，不预报效果。

## 源证据与问题

V16／V22／V23已保存源best原1506 mean分别约0.526217／0.489733／0.490210%FS，但max仍为65.953931／92.611902／62.204719%FS。V24固定原ImageNet参数、以两深层adapter适配，完整98轮源best78原mean／tail10为2.180123／9.578513%FS，roll512为3.952898／14.119596%FS，源拟合明显弱于直接完整语义模型；原462 state与77BN逐轮保持的执行已确认。

本轮检验恢复完整语义学习，并将单点监督改为完整有序分布监督后，能否同时改善源典型精度和严重尾部，再以完整冻结外评检验整体泛化。既有源事实不足以证明严重错由饱和、遗忘或语义瓶颈造成。恢复可训练范围、输出容量和loss共同改变，不能单因素归因。下一方向只依据源证据、静态接口和读数数学确定，不使用目标病例设计标签、阈值或loss。

## 单一模型与部署

只用已有缓存的`mobilenetv4_conv_medium.e500_r224_in1k`，`num_classes=0`，`HF_HUB_OFFLINE=1`。保留完整8434512参数／462 encoder state、conv_head／norm_head和实际semantic1280；`num_features=960`不是输出维度。新Linear1280→256的weight为Normal(std=.001)、bias0，所有原CNN及新head共同训练，BN仅在源训练更新、eval固定，不沿用V24冻结政策。

固定K=256和节点`x_i=i/255`，注册一个FP32节点buffer。实际CPU零前向构造确认head327936参数、完整8762448参数、233个参数tensor与465 state，全部参数可训练。没有hidden MLP、sigmoid point、adapter、FPN、query、teacher、角色／几何分支、router、输出融合或TTA。

`log_distribution(image)`仅执行一次CNN和head，返回FP32 `logp=log_softmax(logits.float())`。`forward(image)`复用此路径，以`p=exp(logp)`输出`Σx_i p_i / Σp_i`。训练每batch也只取得这同一logp，用相同期望作连续误差统计，再算下述loss。两个入口只接收image，不接受y／GT几何／域标识，不缓存带图的p／logits／semantic。全部256质量参与分子或分母；质量分母只消除FP32求和偏差，不筛峰、丢质量或按confidence归一，不clip预测或GT。

`ROI RGB256 → 完整trainable encoder → semantic1280 → Linear256 → 有序logp → 全质量期望reading`。

## 唯一源监督

归一化真值y只在源loss侧使用。其两个相邻节点重心标签的目标CDF直接为：

`Q_j(y)=clamp(j+1−255y,0,1), j=0,…,254`。

对应两热分布总质量1，节点期望为连续y；0与1分别为首／末节点，不环绕，不用argmax量化或Gaussian标签。模型读数不需要GT。

在FP32由同一logp计算`logF_j=logsumexp(logp_0,…,logp_j)`及`logS_j=logsumexp(logp_{j+1},…,logp_255)`，分别使用正向及反向logcumsumexp。唯一loss为：

`L=−mean_(batch,j)[Q_j logF_j+(1−Q_j) logS_j]`。

255个阈值的平均是固定归一化。没有SmoothL1／CE／KL叠加、熵／单峰／confidence正则或loss系数。不能先exp再log，不能通过`log(1−F)`相减、经验epsilon或阈值掩盖错误侧小概率。有限极端logits仍应产生有限累计log与必要梯度。encoder／head用AMP F16，log_softmax、CDF／loss和期望为FP32，不扫描精度／scale。

该loss监督完整有序CDF，同均值的不同分布可以被区分；两热目标的期望保持连续读数，避免固定argmax量化。SmoothL1本来就惩罚远错，不能说CDF天然赋予更大尾部权重。稳定log路径的极端错误梯度是可单独检查的数学性质，不证明旧严重错来自饱和，也不保证实际encoder或外域收益。目标两热分布在非节点位置有固有Bernoulli熵，最小loss不必为0。

V20／V21的256是角度关系网格并构造联合J／几何运输，本轮256直接是[0,1]读数节点，没有校准、射线、A/P/B角色、J或CircularProgressPushforward。V19的四视图C4语义均值也不复用；本轮一次原ROI CNN。普通分布期望与有序监督不作为新理论。

## 固定源合同与停止

完全复用V22完成源22686unique：old18590一次＋roll4096四次，每epoch34974呈现／547批64末30。原1506／24调用＋roll512／8调用，共2018／32。256 OpenCV LINEAR／ImageNet与原contrast／saturation .8–1.2、brightness ±.1保持。不新增源图、增强、名单、GT或seed；原14494／1506／4000及所有fixed cohorts／focus保留。roll512独立layout／ID／group／声明generated seed但sharedHDR；原20096 metadata缺seed，原1506 scene guard保持，不认证完整历史seed链或新域盲测。

fresh AdamW LR3e-4／wd1e-4／clip5／default fresh GradScaler／AMP／seed20262020／train64／val64／workers4。源R仍为`(mean_original+mean_roll+tail10_original+tail10_roll)/4`，normalized full errors、失败error1，tail最大ceil10%分别151／52。最早严格最小R选best，同一R用于Plateau factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8和早停。min60／max160，epoch≥60且epoch−best_epoch≥20停止，counter从1计且LR／resume不重置，不延长上限或按目标调参。

完整保存model／optimizer／scheduler／scaler／RNG／history／totals／init和源合同，严格本轮protocol／config／loss／decode恢复。历史源loss字段明确为CDF log-score，不与旧SmoothL1数值直接排序。last只两源记录，部署NMAE始终由同一分布期望得到。

## 必要新检查与执行

CPU toy仅检查新分布与接口：两热CDF端点／小数／质量／连续期望、左右logcumsumexp与中等值直接公式相符、有限极端错误logits的loss／梯度和方向、同均值不同分布的CDF区别、双峰期望仍落两峰之间、单CNN共路径／B1/B3 eval partial／RGB-only拒GT／必要encoder和head梯度／strict新state。真实模型CPU只做零forward构造，0RGB／0cached权重／0GPU核实actual metadata。完整cached462 state逐项保持在实际源见证首forward前确认，不另跑旧模型。

一次真实源AMP零更新见证：4张已有roll训练0／1图＋两源验证各2张，恰3CNN／8图／1combined backward／0optimizer。训练从一次log_distribution取logp、同期望与唯一CDF loss，两验证各走部署forward；实际head与CNN调用一致。记录semantic1280／256分布、FP32概率质量／CDF／期望、encoder前部／完整neck／新head有限非零梯度、233参数tensor无optimizer不变、77BN源训练各+1随后eval额外0、全部state严格reload及两验证256输入identity。pref唯一back固定loss×128后grad÷128，formal default GradScaler不改不扫描。正式fresh重建，不沿用prefBN。

V24完整ONE核对与规范化归档后，确认实际GPU空闲和本轮代码／配置已冻结，才启动唯一串行pref→source→Industrial→RF100队列。活job只监控queue／currentstep原日志／history／latest及实际父子GPU；不改livecode／model／config／名单／标签或追加模型评估，DataLoader4workers不是额外模型，best不变不是停滞。真实实现fault先原日志和活cmd，保留failure／partial／snapshot，必要实际修复＋针对检查后同output/latest恢复；实质方法变化另实验，不删样本或绕保护。

## 外评、成本与归档

完整源停止锁定best后，各一次冻结Industrial1395／52与RF100151／35。两域不训练、总体统计拟合、BN适配、checkpoint／epoch／早停／LR／训练超参／节点数／loss／decode／阈值／ROI／温度选择，不用目标病例设计结构或标签。推理只有nativeROI RGB/eval，RFscalar离线四点派生，失败完整分母error1，不静默删行或把有效饱和读数误叫invalid。last不外评。

E轮实记train34974E图／547E CNN和backward尝试、val2018E图／32E调用，updates＋AMPskip=547E；外域1546／25／0back／0update，总36992E＋1546图／579E＋25调用。每次head实际执行，分布／CDF额外计算、time／peak另记，一个CNN不等于相同成本。pref单列，旧源生成不重复算新费用。

完整结果后仅ONE新鲜独立CPU保存数字／执行核对，规范化显式allowlist归档现有main。报告mean／Acc1/2/5／median/p90/p95／tail10／max、>1/2/5新增救回、组／zero／非zero／近zero、ordinary1495/1497/severe6和固定focus；roll512单报不补评旧模型。rawrun／fullcontracts／sourcearrays／RGB／weights／H/C/XYZ/U/M／bbox／features／privatepaths私有。真人授权单轮完成后持续推进新独立源域表示迭代，不重跑completed job。

## Limitations

全质量期望仍平均多峰，并不输出某个必然真实模式；proper CDF loss不保证MAE／tail／max或外域校准最优。训练范围、容量和监督共同改变，不作单模块因果或语义遗忘证明。共享HDR源验证不是真实域盲测，单seed、两个已查看开发外域、旧GeoPRR历史pixel／完整权重链未重新认证的边界保持；不报SD、显著性、新盲测或新理论。完整正负结果与新增大错均保留，不自动升级模型。

文献仅作已有机制背景：[视觉标签分布与期望回归（IJCAI 2018）](https://www.ijcai.org/proceedings/2018/0099.pdf)采用不同Gaussian目标和双loss；[深度有序分类的单峰概率分布（ICML 2017）](https://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf)使用本轮未采用的单峰参数化。二者不证明本任务zero-shot或尾部收益，本轮不复制其完整方法。
