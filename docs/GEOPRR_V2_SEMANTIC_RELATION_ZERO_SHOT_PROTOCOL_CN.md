# 完整语义条件关系读出：独立zero-shot协议

2026-10-23。v20按下述冻结协议完成60轮与两套冻结外评，source-best为第16轮：源NMAE 0.283315%FS，Industrial 29.257524%FS，RF100 16.224095%FS。相对v16源均值降低46.1601%，但两个外域均退化，本配方不作为默认模型、不延长训练。一次独立CPU核对54062项通过、0失败；见[完整结果](data/geoprr_v2_semantic_relation_20261023/completed/report_cn.md)。执行代码9494af66d04ea3ce0da68f7fd8d579c154814087。目标域未用于训练、适配或训练超参数选择，全部旧模型、原G1／anchor／65轮和负结果保留。

## 可检验问题

v18用局部关系作训练辅助，但部署只保留semantic直接标量；它改善源误差和部分RF100参考型错误，同时使Industrial和RF100严重尾部回退。v19在encoder之后池化四方向semantic，源预测接近常数，尚未建立有效源拟合。前者提出部署读出是否保留关系的问题，后者提出在汇总前保留局部角色关系的问题；这些结果不证明任何C4约束都不可能有效，也不证明本轮结构必然泛化。

本轮检验：保留完整ImageNet语义作为校准上下文，由同一个CNN的多尺度空间特征预测参考条件角色联合分布，再直接用该联合分布读数，能否在源域学到有效读数，并改善冻结模型的跨数据集表现。这里组合已有全质量关系读出和完整语义条件，不作新算法或理论认证，不加入direct point、融合MLP、router、旋转轨道或teacher。

## 唯一部署通路

`256 ROI RGB → 一次完整MobileNetV4 → semantic1280 + 四尺度FPN64 → 语义条件center／SPD与参考角色联合J → 原CircularProgressPushforward → reading`。

encoder为已有缓存的`mobilenetv4_conv_medium.e500_r224_in1k`，`num_classes=0`，保留conv_head／norm_head。实际semantic输出为1280维；报告的num_features960属于neck输入，不能当作最终embedding。只使用缓存ImageNet fresh初始化，HF_HUB_OFFLINE=1，不装入任何旧任务checkpoint，不下载新foundation。

一次共享encoder前向同时取得blocks.0／1／2／4的stride4／8／16／32特征，通道48／80／160／960，进入64通道FPN。沿用256 bins、decoder_dim64、token_dim16的local ReferenceRelationStateHead；context首层输入从原三尺度汇总192维改为完整semantic1280，使预测center／SPD实际受完整语义条件控制。后续径向采样、参考条件角色和相对联合的定义保持原有语义；没有最终Linear1280→1直接标量头。

A为参考起点，P为半径1的虚拟dial-plane状态点，B为量程终点；P不是实体针尖。模型保留共同A条件下的P／B关联，形成相对角差d与跨度s的联合J。J不是delta与span两个边缘分布的乘积，也不先将角色压缩为单个角度或坐标。部署调用不变的CircularProgressPushforward，将J的全部质量运输到归一化读数分布并取得reading，保留围绕跨度一半的有符号展开、端点映射和零跨度均匀运输。

训练和eval的forward均只接收RGB。center／SPD、rays、角色与跨度全部由模型预测；GT H、真实span、解析U／M和角色标签只供源loss使用，不能进入部署读出。全encoder、semantic context、FPN和角色参数均训练，BN只在源训练更新，验证和外域固定eval，不拟合目标总体统计。

实际CPU构造核实encoder为8434512参数／462 state，FPN与关系头为327540参数，完整8762052参数／502 state，共262个参数tensor，其中31个属于非CNN模块；相比v18辅助结构增加69632参数来自context输入变化。完整502项checkpoint用于训练恢复与严格eval加载；不把预期计数提前写成实际结果。

## 源监督与读出兼容性

源人口沿用18590 unique训练图：原14494与4096组内状态，复用v11已有512张真实端点替换图；原1506验证和4000 officialtest不变，不新增渲染；端点0／1指归一化量程起点／终点，0不必是绝对物理读数零。256 RGB／OpenCV LINEAR／ImageNet归一化、原contrast／saturation 0.8–1.2与brightness ±0.1光度策略保持。无配对重复曝光、ROI扰动、MixStyle或新增seed。

监督沿用v18已完成解析合同：H／span和虚拟A／P／B角色、64网格解析进度field U与ring／arc mask M。RGB人口、物理y、路径和bbox基础子合同与v16／v18相同；新增部署关系读出不等于完整模型合同相同。原有向C4／ordered ticks／校准限制保持，不剪裁虚拟P、不改端点标签、不删行。没有Gaussian Q或Q×M×U组合GT读出，不将旧Q支持／readout审计转写成本轮新保证。

现成的一次源CPU数学检查用v18实际FP32两热角色标签，按共同A支持角构造条件联合：

`J[d,s] = sum_a hA[a] * hP[(a+d) mod256] * hB[(a+s) mod256]`。

求和使用A两热标签的支持角，保留同一个A条件，不能替换为两个相对边缘的独立乘积。随后调用原CircularProgressPushforward。全18590行／291批已检查，11.688346秒、0 RGB解码／CNN／反传／更新；均值0.002771582%FS、max0.430768728%FS，0行超过原1%FS兼容界限，最大联合／读数分布质量偏差2.384186e-7，零跨度质量均为0，102行存在角色两热环绕。端点0／1各128行，max分别0.428884476／0.430768728%FS。规范化记录见[源映射兼容性](data/geoprr_v2_semantic_relation_20261023/source_role_mapping_feasibility.json)。

该记录只表明理想源角色标签与不变读出的兼容性；不加载模型，不证明角色可学习或外域有效，也不重复旧几何GT／教师／全源解析检查。原1%FS仅作兼容性参照，没有放宽阈值或构造新的强制通过标准。端点处outside_arc_mass约0.999997时，理想reading仍可准确为1；因此不能将该质量作为准确率硬门控。

唯一loss与v18系数及分解相同，primary替换为实际部署关系reading：

`SmoothL1(relation_reading,y,beta=0.05) + 0.1 * (role_NLL/(3*log(256)) + field_L1)`。

不增加CDF、额外角差／跨度损失、融合或confidence门控；0.1不扫参。辅助角色／场在本轮同时支撑实际部署读出，不再描述成只在训练运行的几何结构。

## 固定源训练与冻结外评

独立fresh AdamW，weight_decay1e-4、gradient_clip5、seed20262020、AMP、train64／val64／workers4，固定60完整epoch。LR同v16：epoch1–30为`3e-4*(1+cos(pi*(e-1)/30))/2`；31–60为`3e-5*(1+cos(pi*(e-31)/30))/2`。不从v16／v18／v19继续训练，不机械延长旧预算，不扩seed或做参数矩阵。

每轮仅按完整1506源验证的部署reading NMAE选择最早严格最小best；last60仅记录源结果。全部60轮完成后，冻结source-best，各一次Industrial1395／52组和RF100151／35组。目标标签只用于计分，RF100 scalar仍由四点离线派生；原生ROI RGB是唯一推理输入，不使用目标GT几何或额外GT裁剪。

Industrial／RF100不得用于训练loss、总体特征统计拟合、BN适配、早停／checkpoint／epoch、阈值、ROI、温度或训练超参数选择。模型产生无效scalar或输入失败时，完整原图分母保留normalized error1，分别记输入与scalar失败，不以过滤失败获得成绩。两域为已查看开发人口，不称新盲测。

计划每轮18590训练图／291次CNN与关系读出、1506验证图／24次调用；60轮为1115400训练图／17460反传尝试、90360验证图／1440次调用，外域1546图／25次调用，总1207306图／18925次CNN。关系读出调用分别17460／1440／25。实际有效updates与AMP skips之和须为17460，训练、验证、外评时间、process／full时间、峰值内存及resume分别实记，不报计划已完成。部署虽只有一次CNN，FPN／角色／全质量读出仍增加计算，不能据相同CNN调用数声称与direct推理等成本。

## 实现、运行见证与归档

入口`experiments.run_geoprr_v2_semantic_relation`，独立输出`runs/geoprr_generalization/iteration_v20_semantic_relation_20261023/source_only_training`。执行代码9494af6按冻结配置完成唯一源队列。4项新CPU检查已通过（0.205秒），实际模型参数构造已核实。真实源RGB见证也已通过：4张训练源0／1图加2张原256验证图，共2次CNN／6图／1次combined反传／0optimizer；完整462项缓存ImageNet初始encoder相同，1280上下文实际输入与各必要模块梯度有限非零，77个BN counter源+1／eval额外0，完整502 state严格加载，验证输入256 identity，纯RGB接口拒绝GT几何。见[实际运行记录](data/geoprr_v2_semantic_relation_20261023/runtime_preflight.json)。复用v18解析源合同记录和上面的理想联合映射记录，不重新执行旧GT／教师／完整数学检查。

实现后只做新结构必要的针对性CPU检查及一次实际源RGB零更新见证：真实0／1源图、单CNN多尺度和semantic1280确实进入context、RGB-only接口、部署reading及loss有限、encoder／neck／FPN／context／role梯度有限且实际相连、源BN更新／eval固定、256验证输入identity、配置精确和完整实际state严格reload。见证0optimizer，正式训练fresh重新初始化，不沿用见证BN状态；实际参数与state以构造核实记录为准。

只有旧队列complete、父子退出、GPU实际空闲、协议／配置／预算／版本冻结及新源见证通过后才启动唯一队列，串行source60→Industrial→RF100。以pipeline PID／queue current_step、原日志、history和latest判断运行，不以不变的best时间判断停滞，不在阶段交接GPU短暂为空时重复启动；DataLoader workers不是新模型。活job期间只监控，不改model／code／config／人口／标签或追加GPU评估。

保存完整model／optimizer／scaler／RNG／history、初始化与合同摘要、源best／last各1506逐图、两外域完整逐图／组／summary和全部真实成本。真实实现故障先读当前step日志及活命令，保留failure／partial／snapshot，在同output从latest恢复；方法、配置或人口实质变化另立独立实验，不删样本或绕保护。

整个阶段完成后仅一次必要独立CPU数值与执行核对，再按各域连续NMAE／Acc、zero／nonzero／nearzero、组、ordinary1495／1497、severe6、3172／2810／3663／9323／8195及sync9／RF41、tail／max／救回新增与成本比较v16、v18、v12及旧GeoPRR保存参照，不重跑旧模型。仅本任务代码和显式规范化allowlist可推现有main，不建branch；RGB、权重、bbox、H／C／XYZ／U／M、特征矩阵、raw run／data_contract和private trace保留私有。

实际启动父421867／初始源训练子421874，2026-10-23 09:41:48 UTC核实精确命令匹配，唯一GPU进程为421874；首批64源图loss0.334367186、梯度范数有限。这只证明任务正常执行，不是准确率结论。见[启动快照](data/geoprr_v2_semantic_relation_20261023/launch.json)。

三阶段均正常exit0，最后RF100阶段于2026-10-23 10:32:37 UTC结束；旧父421867与源训练子421874已退出，完整结果及实际成本已归档。

## Limitations

本轮同时改变语义条件输入和部署读出，不能单因素归因某个模块。理想标签映射误差不等于可学习表示的误差，量化、预测center／SPD／角色错误和源监督负迁移仍可能发生。v18／v19结果仅提供可证伪问题，不构成本方案有效证据。源更准不保证外域更准；单seed、已查看开发域不支持SD、显著性或普适泛化声明。旧GeoPRR原生输入、训练史及历史像素／完整权重链不同，未重新认证。全部旧失败保留，无收益不自动升最终模型、不延长60轮或追加模块。
