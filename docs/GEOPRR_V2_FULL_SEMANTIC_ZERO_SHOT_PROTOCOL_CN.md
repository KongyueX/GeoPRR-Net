# MobileNetV4完整语义特征与直接进度：独立源域训练

2026-10-23。用户要求继续提高zero-shot表现，不限定旧GeoPRR方法，Industrial与RF100不得用于训练。本轮为独立新模型，不延长v15的20轮缓存线性头，也不恢复旧65轮任务。状态：配方冻结，代码及新256源输入运行检查准备中，尚未启动正式训练。

## 证据与本轮问题

v15的冻结几何MobileNet特征＋线性sigmoid探针在源验证本身就表现很差：source-best19为17.178667%FS，原几何读数0.063613%FS；外域direct为39.015321／38.410688%FS。完整20轮、源train-only统计、真实更新和新读出均经一次独立CPU核对142946项通过。这不是“源域已学好、只在外域失败”，不能据此否定端到端直接进度表示，也不继续延长缓存线性头。

旧GeoPRR的clean优势几乎已经存在于Raw base标量读出；但它的EfficientNet表示曾以直接进度目标端到端训练。当前角色几何CNN训练目标和GAP特征的线性可读性不同，移植一个输出层不足以复现旧Raw表示。

本地离线CPU确认，完整`mobilenetv4_conv_medium.e500_r224_in1k`分类模型保留`conv_head:960→1280`和`norm_head`。`features_only`路径没有这6个ImageNet state tensors／1231360个参数。完整encoder去分类器后为8434512参数／462 state；`num_features=960`描述neck输入，实际分类embedding为1280，不能据该字段误建960维point head。

本轮检验完整预训练语义通路＋端到端直接进度监督是否比现有几何专门化表示更适合跨域。它同时改变初始化历史、读出、分辨率和训练目标，不作为单一neck或骨干因素的因果认证。

## 模型与初始化

使用同一既有ImageNet缓存，`timm.create_model(name, pretrained=True, num_classes=0)`保留完整encoder／global pool／conv_head／norm_head。新增`Linear(1280,1)`，weight以seed20262020、std0.001初始化，bias0；点预测为sigmoid，输出归一化进度。没有虚拟角色、预测H、center、SPD、角差／跨度运输、expert、router或不确定性后验。

预计总参数8435793、state464，其中462个encoder state来自已缓存ImageNet，1281个point参数／2个state为新初始化；实际以构造和运行记录为准。首次初始化必须`HF_HUB_OFFLINE=1`，不得下载新foundation权重。恢复与冻结评估用`pretrained=False`构造，再严格加载本轮完整checkpoint，不重新引入ImageNet或其他模型参数。

## 源数据与输入

训练使用既有端点替换源18590张unique图，原验证1506张；源ID／group分离，原manifest和全部旧资产保留。原14494图与4096组内状态图、其中512个真实端点替换完全沿用已完成源阶段，不新增渲染或物理GT检查。不用donor／recipient重复曝光。

仅以归一化源读数y监督，数据入口`require_geometry=False`。H／field／mask／虚拟角色不计算、不输入模型、不进入loss；源标注bbox仅用于既有ROI裁剪。训练采用原有光度增强：contrast与saturation U[0.8,1.2]、brightness U[-0.1,0.1]，不加新几何增强或其他模块。

本模型声明输入为256 RGB、OpenCV LINEAR直接resize、ImageNet归一化；源验证和两外域使用同一256配置。它与旧MobileNet几何768配置有意不同，不能将新入口identity写成与旧768张量相等。`output_size=64`只是沿用数据类的配置字段，几何关闭后不生成空间监督。训练RGB损坏即停止，不删样本；验证与外域输入／预测失败保留并记normalized error1。

## 冻结训练预算与选择

全模型源训练固定60完整epoch，batch64、validation batch64、workers4、seed20262020、AMP、fresh AdamW、weight_decay1e-4、gradient_clip5。唯一监督为`SmoothL1(prediction,y,beta=0.05)`。

学习率按旧Raw实际记录的两段配方提前确定：epoch1–30为`3e-4 * (1+cos(pi*(epoch-1)/30))/2`；epoch31–60为`3e-5 * (1+cos(pi*(epoch-31)/30))/2`。各轮使用的LR直接记录，恢复按epoch重建确定配方；不按外域调LR或轮次。

每轮18590训练图／291批，1506源验证图／24批；60轮训练1115400张CNN图像、17460次尝试更新，验证90360图／1440调用。有效update／AMP skip按实际记录，不绕过非有限loss或坏数据保护。源验证全1506 NMAE选择最早最优best，固定last60独立保留；即使早轮best，也完整执行冻结60轮。

每次保存模型／optimizer／scaler／RNG／history／累计曝光，故障后保留日志和partial，在同output从latest恢复。该独立预算不冒充与旧4轮／709批或旧GeoPRR 12818图60轮完全匹配的训练史。

## 目标域隔离和冻结推理

Industrial1395／52组与RF100151／35组不参与梯度、损失、早停、checkpoint、特征统计、BN适配或超参数选择。只有60轮源训练完成、source-best冻结后，才各运行一次完整评估；不测试多个epoch后挑外域最好者，不混合v15／旧GeoPRR预测。

外域采用原生ROI RGB、相同256 LINEAR／ImageNet、eval／AMP、batch64／workers4，完整失败分母error1。模型只接收image，量程和target仅用于计分；RF100标量由四点离线派生，照实披露。不用目标GT H／center／SPD／针尖、目标BN、ROI／温度／阈值／TTA或标签修正。

两外域合计1546张／25次CNN调用；完整正式阶段计划1207306张CNN图像／18925次调用。hook分别记录训练、源验证和外域实际forward，CNN反传、有效optimizer update与AMP skip分开报告。必要的新源输入运行检查单列，不算正式成绩。

## 执行和判断

入口`experiments.run_geoprr_v2_semantic_direct`，独立输出`runs/geoprr_generalization/iteration_v16_full_semantic_direct_20261023/source_only_training`。启动前必须v15 complete／父子退出、实际GPU空闲、项目环境warm reuse、新协议／CONFIG／版本冻结。只使用项目venv和已有GPU，单队列串行，不新增seed或SyncG officialtest。

针对CPU检查覆盖full encoder保留semantic neck、point形状／梯度、四个LR端点、全分母scalar失败规则。真实源RGB检查确认新256输入合同一致、完整ImageNetencoder保留、1280embedding／标量输出、所有训练分支梯度有限且有非零信号，0 optimizer；不复跑旧GT、teacher或旧完整tests。

完成后收齐config／protocol／初始化记录／run／history／best与last源逐图／completed／两外域逐图和真实成本，只做一次必要独立CPU数值／执行核对。报告source／Industrial／RF100 NMAE、Acc、zero／nearzero／nonzero、组、tail／max、ordinary1495／1497、严重6和3172／2810／3663／9323。比较旧GeoPRR历史、同像素当前几何family及v15探针，但保留各自训练预算和输入配置差异。

## 限制

zero-shot在这里指0目标训练／适配；两外域已被查看并启发设计，不能称全新盲测。此处并未认证历史旧GeoPRR的逐像素同一性，也没有补齐其私有权重链；旧source-only与监督OOF成绩不混用。该模型同时改变表示、读出、分辨率和目标，若有收益，也不能只归因于一个neck；若无收益，保留负结果，不用目标数据补训。单seed不作显著性或普适泛化声明。原模型、全部失败／负结果与冻结名单保留，RGB、特征、权重、raw H／C／XYZ和private trace不上传。
