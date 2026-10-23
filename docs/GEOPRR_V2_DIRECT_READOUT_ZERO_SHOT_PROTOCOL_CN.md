# MobileNetV4独立外观读出：源域训练与冻结外域评估

2026-10-23。用户最新要求：继续提高泛化，不限于旧GeoPRR方法；Industrial和RF100不用于训练。本轮先检验一个结构问题：在同一个冻结MobileNetV4表示上，恢复独立、非循环的RGB进度读出，是否能缓解现有几何角色读出的外域大错。状态：完整源缓存、20轮head训练及两冻结外评已结束，独立CPU数值与执行核对142946项通过；见[完整负结果](data/geoprr_v2_direct_readout_20261023/completed/report_cn.md)。

## 已有证据与问题

旧GeoPRR seed20262020的clean历史同人口NMAE为Industrial11.643618／RF1006.607079%FS。其内部保存诊断为：

| 外域 | base | polar | relational | 最终输出 |
| --- | ---: | ---: | ---: | ---: |
| Industrial1395 | 11.642999 | 11.741234 | 11.642999 | 11.643618 |
| RF100151 | 6.569962 | 6.696136 | 6.569962 | 6.607079 |

两套clean人口的全部行都属于relation_invalid，原入口对clean禁用SARN关系；relational与base相同，最终融合均值也未优于base。不能将该优势直接归因于透视整流、polar或router。旧base在Raw末层图像特征上作全局平均池化、线性projection和sigmoid，再构造保持该均值的posterior。证据见[旧内部诊断](data/official_syncg_fulltrain_20260908/geoprr_diagnostics.json)、`experiments/evaluate_remstnet_real_domains.py`、`remstnet/model.py`与`experiments/a15_2_mett.py`。

当前ReferenceRelationReader只保留backbone／lateral／refine，未保留普通V2的appearance head。所有读数经过预测center／SPD、径向取样、三个虚拟角色联合识别及角差／跨度映射。旧方法与当前方法的编码器、训练史和预算不同，现有比较不能证明单一读出机制的因果优势，但直接读出是一个具体可检验的差异。

旧主模型Raw锚点的实际记录为12818张候选训练图、60轮、12058次有效更新；其完整源池16000中另3182张用于router。独立EfficientNet全16000基线不等于旧GeoPRR锚点。当前可执行旧loader的单调校准默认identity，未找到历史主结果启用knots的记录；不将未知校准解释为优势来源。

现有保存样本也不支持按单个置信度阈值回退：RF119在低outside_arc_mass及高reference_max时仍严重读错，Industrial72的outside质量很高却准确。因此本轮先不引入可靠性router，只保留可分开检验的三种读出。

## 唯一冻结表示与新增读出

完整初始化为原v6 feedback best48、seed20262020、原生global模式，路径`runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best.pt`。全部496 state严格加载，总参数7461060；CNN、decoder、几何角色头和所有BN buffers保持eval且不训练。本轮不是恢复ImageNet表征，也不是新端到端训练。

在该模型同一次前向中，通过只读hook取backbone最后一个实际特征层，作GAP得到向量f；不改变原几何输出，不增加第二个CNN前向。仅用源训练特征估计每通道mean／std，`z=(f-mean)/max(std,1e-6)`。该固定标准化仅改善线性拟合的数值条件，不增加线性函数容量，不使用验证或目标数据统计。

新增直接进度为`sigmoid(w·z+b)`，w、b初始化0。新参数仅D+1个，D由实际原生末层通道数确定；mean／std作为固定buffer保存。没有新MLP、GT几何输入、外域校准、后验置信度声明或自适应路由。

用同一冻结模型和同一新head记录三个输出：

1. native geometry：原几何角色读数，作为必要同一次前向中的参照；
2. direct：新增非循环外观标量；
3. fixed half：原几何标量与direct各占0.5的预先固定均值。

不根据外域结果改变权重、选择readout或重新训练。直接头只按源验证选择一次checkpoint；half使用同一head，不单独挑选epoch。三个输出不是三次CNN推理，也不将已有旧GeoPRR预测或外域标签作为teacher。

## 源数据和新头预算

训练源为已完成的端点替换manifest，18590张unique RGB：原14494训练图及4096组内状态图，其中512张使用已验证真实端点替换。原验证1506保持不变，源训练／验证ID严格分离。源框用于原生ROI准备；head只接收RGB产生的GAP特征，监督只用源归一化读数y，不将H／center／SPD／针尖送入head。源精确0表示量程起点，不必为绝对物理0。

冻结表示仅抽取一次：augment=False、原768／192、ImageNet、AMP，训练cache batch32，验证cache batch16，workers4。缓存18590训练＋1506验证共20096张CNN图像、676次调用；源特征、权重、manifest及原始几何只私有保存。缓存结束逐项比较冻结原生state，含参数与BN buffers，确认其全部保持不变。

直接head在CPU缓存上训练固定20轮，batch256、seed20262020、AdamW LR1e-3、weight_decay1e-4、余弦min_lr1e-5、normalized L1。每轮18590条unique特征，73个batch；完整预算371800次训练特征呈现、1460次head backward／optimizer更新，CNN backward／optimizer均为0。该曝光不同于旧709批donor／recipient训练，不冒称预算匹配的全量微调控制。

每轮用原1506源验证缓存计算direct NMAE，严格较低时保存best，平分取早轮；固定last20另存。统计量不拟合验证特征。外评仅使用这一个source-best head，不评估多个epoch或使用外域挑选head。新head学习能力不足的负结果，不能直接否定端到端训练直接读出或更丰富源表示的可能性。

## 冻结外域评估

Industrial1395／52组和RF100151／35组完全不进入训练、特征标准化、适配、checkpoint或超参数选择。只有源训练结束、best head冻结后，才启动两套外域评估；使用原生ROI RGB、原768／192、ImageNet、eval／AMP、batch16、workers4。RF100标量沿用四点离线派生，模型输入不包含目标GT几何。无目标BN、ROI、温度、阈值、TTA或标签修正。

三个readout在同一次前向分别计分，完整分母与normalized error1规则保持。half要求两个标量均有效，不利用置信度静默丢弃分支。新增外域1546张CNN图像、98次调用；正式全阶段合计21642张CNN图像／774次调用。实际hook计数和head更新分别记录，不能把仅CNN无反传写成整个阶段0反传／0更新。必要2张真实源RGB运行检查单列，不冒充性能结果或正式成本。

## 执行与判断

入口`experiments.run_geoprr_v2_direct_readout`，独立输出`runs/geoprr_generalization/iteration_v15_direct_readout_20261023/source_only_probe`。固定代码／CONFIG／协议、独立锁／PID／步骤日志；旧v13／v14 complete且父子退出、实际GPU空闲、新输入运行检查通过后再启动。只复用项目venv和已缓存权重，不下载基础模型、不扩seed、不重跑SyncG officialtest。阶段依次为源cache、CPU head fit、冻结Industrial／RF100；不并行占用其他GPU。

必要检查只覆盖新功能：4个CPU针对测试检查源train-only统计、head有限梯度／有界标量／half算术及单CNN hook不改native输出；新真实源输入检查确认完整原生加载、GAP通道、新head有限梯度、native state保持及验证输入identity，0 optimizer。完成后收齐缓存元数据、训练history／best／last、两外域三readout逐图／组／摘要与真实成本，只做一次独立CPU数值／执行核对，不重复旧G0、GT、teacher或旧完整测试。

主要判据是同一冻结MobileNet表示上的外观与几何读数是否互补，以及direct或预先固定half是否改善完整外域均值／尾部；源均值、ordinary1495／1497、原严重6和3172／2810／3663／9323同时照报，不恢复零新增错一票否决。若direct不能在源上学习，先解释为固定表示＋线性头的容量检验；若源好而外域差，说明该结构仍未解决分布偏移。若只有一域有收益，保留该限制，不升级为普适方法。

## 限制

zero-shot在这里明确为0目标域训练／适配；这两套外域此前已被查看，设计受诊断启发，不能称全新未触碰盲测或统计显著性。历史旧GeoPRR像素级同一性仍未重新认证，旧权重链当前不可重放；本轮不是其严格复现或单因素骨干比较。冻结特征的新head探针不等同于完整端到端混合模型，也没有增加背景多样性。源cache不使用光度增强，缓存特征训练的预算和作用范围独立报告。原模型、全部旧失败／负结果、14494／1506／4000与冻结名单保留；公开仅显式allowlist中的代码和规范化结果，原图、特征、权重、H／C／XYZ与private trace不上传。
