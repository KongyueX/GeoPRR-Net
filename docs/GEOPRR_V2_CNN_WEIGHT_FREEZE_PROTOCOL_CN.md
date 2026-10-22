# MobileNetV4固定CNN权重的端点覆盖匹配实验

2026-10-23。状态：v12完整核对通过，独立协议／配置与实际输入运行见证通过后，两臂串行训练于北京时间2026-10-23 07:08:12启动。承接用户授权的跨数据集自主迭代，复用v11已完成的128组／512张真实端点源图，不新增渲染或数据替换。全部旧模型、失败和负结果保留。

## 问题与选择依据

检验固定既有CNN权重、只训练后续解码和读出部分，能否保留端点覆盖的外域收益并减少源域尾部损伤。v11正常BN端点版best1的Industrial／RF100为17.353598／9.643770%FS，较匹配v9正常BN参照改善4.75%／5.35%；源NMAE却从初始化0.063613升至0.068488%FS，2810达到9.506392%FS，新增超过5%FS的严重错误。固定BN统计的v11臂没有取得同样的两域增量。因此本轮继续更新源BN统计，单独约束CNN参数更新。

v12完整数值／执行核对通过，local端点版best1的源NMAE为0.083392%FS，高于local57初始化的0.077123%FS；最大误差18.6776%FS，高于初始化的5.90885%FS。其Industrial／RF100为20.5391／7.65916%FS，对应原源续训参照为23.5072／9.55355%FS；固定第4轮也保留两域收益。选择feedback与local两个已有家族，是为了检验同一训练约束在两种既有模型中的表现；两者锚点训练史不同，不将跨家族差异归因为单一归一化因素。

文献提供的是可检验动机。Kumar等在预训练特征较好且分布偏移较大的分类场景中发现，完整微调可能损伤域外表现，并提出LP-FT；这支持检查参数更新范围，而不证明本任务应冻结CNN。[Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution，ICLR 2022](https://arxiv.org/abs/2202.10054)。Li等的AdaBN通过改变BN统计进行域适应，提示权重与统计是不同的可调对象。[Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)。本轮只更新源训练BN统计，不使用目标数据估计统计。

## 两个家族各自的匹配对照

| 新训练臂 | 完整初始化 | 原生读出与model配置 | 已完成的对应参照 |
| --- | --- | --- | --- |
| `feedback_cnnfixed` | v6 feedback best48 | `geoprr_v2_reference_feedback_source_only_v1`；原五字段，含`reference_normalization=global` | v11 `endpoint`正常BN端点臂 |
| `local_cnnfixed` | v5 local joint best57 | `geoprr_v2_reference_relation_source_only_v1`；原四字段，默认local | v12 `local_endpoint`正常BN端点臂 |

锚点分别为`runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best.pt`和`runs/geoprr_v2_tail/iteration_v5_reference_relation/readers/joint_s20262020/best.pt`。两臂均严格加载全部496个state，保留7461060总参数；不重置CNN，不恢复ImageNet权重。原四个共同model字段为`bins=256`、`decoder_dim=64`、`token_dim=16`、`coupling=joint`，local配置不额外加入显式归一化字段。

唯一新训练约束为`freeze_backbone_weights=true`：固定CNN的全部参数，包括BN affine；`freeze_bn_statistics=false`，源训练时BN running统计仍正常更新。lateral、refine和head继续训练，实际可训练参数数、CNN权重保持及BN buffers变化分别记录。该配方不使特征计算固定，因为BN统计仍会变化。两臂共用新协议`geoprr_v2_endpoint_backbone_weight_freeze_source_only_v1`，没有新增网络模块、loss或参数。复用上述对应端点控制，不重训v11／v12，不把v9的不同源数据控制当作本轮唯一因素参照。

## 源数据与冻结预算

两臂均使用`rendering/generalization_endpoints_v11_20261023`，精确复用[端点数据合同](GEOPRR_V2_ENDPOINT_GENERALIZATION_PROTOCOL_CN.md)：原1024组计划的前128组，512张真实RGB状态为`[0, 0.005, 0.02, 1]`，替换相同场景／render seed／C4顺序的旧quartile图。其余18078行、验证路径与合同、原阈值和边映射保持原样。0为归一化量程起点，未必为绝对物理读数0。旧图与manifest保留；本轮新增渲染为0，不增加端点比例或背景多样性。

各4个完整epoch，seed20262020、fresh AdamW、共享LR1e-5、head LR5e-5、min_lr1e-6、warmup0、weight_decay0.01、gradient_clip5、AMP。CNN参数固定后不进入优化器，lateral／refine仍使用共享学习率。train batch32、validation batch16、8 workers、768／192输入与输出。原三项精确`roleNLL/(3*log(256)) + fieldL1 + CDF`，系数各1；不加ROI增强或MixStyle，原配置中的MixStyle p0.5／alpha0.1因模块列表为空而不执行。

每轮18590 unique、22686 donor、8192 recipient、30878训练CNN图像、709批、1506验证图。每臂计划2836批、90744 donor、32768 recipient、123512训练CNN图像和6024验证图；两臂训练图像预算合计247024，验证12048。有效update、AMP skip、前反向实际次数、时间、峰值内存和可训练参数按真实产物记录。冻结CNN权重仍需CNN前向，不把固定参数数或计划批数冒充实际训练成本。v11渲染成本另列并复用，不再次记作本轮新增成本。

## 选择、外评和执行条件

只按各臂4轮完整源1506 NMAE选best-source，另保存last4；初始化锚点不自动加入该次best选择。分别评估完整Industrial1395／52组和RF100151／35组；若best为epoch4，复用同一权重的外评，避免重复GPU。原生ROI RGB、eval、ImageNet归一化、768／192及对应原生Reader严格加载保持一致；不按外域选择epoch、阈值、ROI、温度或TTA，不使用目标label／GT H／center／SPD／针尖作训练或推理输入。全分母保留失败并按error1计，RF100标量由四点离线派生，照实披露。

独立入口为`experiments.run_geoprr_v2_generalization --study cnn_weights_fixed`，输出`runs/geoprr_generalization/iteration_v13_cnn_weight_freeze_20261023/matched_training`。队列记录每臂原生配置与锚点映射、共同源数据、版本、PID、独立锁和代码快照。正式运行要求v11、v12和端点源队列全部complete，对应父子controller已退出，实际GPU空闲，并已完成v12唯一一次必要CPU数值／执行核对。旧队列、协议和代码快照不覆盖。

随后对精确两份新配置执行一次实际输入运行见证：完整496 state严格初始化；512精确替换和验证identity成立；源RGB loss／梯度有限；CNN参数包括BN affine均不更新，lateral／refine／head有可训练梯度，源BN统计仍更新；外评Reader按新协议及原生家族严格加载。结果写入`diagnostics/geoprr_generalization_v13_cnn_fixed_preflight_20261023/result.json`，要求`status=passed`、精确configs、`counts.optimizer_updates=0`、`validation_input_exact_identity=true`；不预先生成通过记录。小批见证不替代正式性能或成本。

实际运行见证已通过：2张独立真实端点RGB、2 CNN calls／4图／2 backward／0 optimizer；两臂全部496 state严格初始化。每臂固定CNN参数7203152个，lateral／refine可训练228096个、head可训练29812个，合计257908个可训练参数；228个BN buffers中152个running统计变化、76个计数器各前进1。两臂全部decoder／head梯度有限且非零，CNN参数及BN affine无梯度并逐项不变；2个验证输入identity、外评Reader严格加载及原生global／local模式均通过。见[运行记录](data/geoprr_v2_generalization_20261023/v13_runtime_preflight.json)。代码1b7e43b，父394447、初始feedback子394452及唯一GPU进程已核实，首批32 donor＋11 recipient、loss 0.0909534320有限；见[启动记录](data/geoprr_v2_generalization_20261023/v13_launch.json)。

两臂串行训练和外评，live期间只监控，不改模型／配置／名单／标签或额外启动GPU。完整阶段后收齐protocol、config、run、history、best／latest逐图、completed及两外域逐图与成本，只进行一次必要独立CPU数值／执行核对。分别与本家族初始化及对应v11／v12端点控制比较连续误差、Acc、零位／近零／非零、组、尾部／最大误差、源普通1495／1497、严重6、3172／2810／3663／9323和救回／新增严重错误。无增量则保留负结果并结束该配方，不延长4轮或旧65轮。

公开仅显式allowlist中的代码、规范化标量／逐图、协议和计数；RGB、权重、raw run／data_contract、H／C／XYZ、像素和private trace保留private。原14494／1506／4000、旧名单、G1／anchor失败、配对守恒负结果及全部旧模型保留。

## 限制

Industrial和RF100已被反复查看用于开发，单seed短训练不构成新盲测、显著性或算法创新认证。两个不同历史锚点仅提供家族内匹配证据；源精度、尾部损伤与外域均值必须同时报告。本轮CNN已针对本任务训练，后续为非线性decoder和head，源BN统计继续变化；这不是论文中的线性探测或LP-FT，也不证明恢复ImageNet通用表征。AdaBN引用只说明统计与参数可分开研究，本轮不做目标域统计适配。两个域同时改善、源尾部保留或普适泛化都不是预设结论。
