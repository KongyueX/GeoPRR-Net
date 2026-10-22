# MobileNetV4 local读出端点覆盖匹配实验

2026-10-23。状态：协议和两份配置已冻结；完整锚点／输入／梯度运行见证通过后，两臂各4轮的串行训练队列已于北京时间2026-10-23 05:53:31启动。承接用户授权的跨数据集自主迭代，复用v11已完成的128组／512张真实端点源图，不新增渲染。旧模型、源图、失败和全部负结果保留。

## 问题与选择依据

检验端点覆盖在local参考归一化模型上是否仍有实际收益。现有family比较中，v5 local best57的RF100 NMAE最低，为9.159439%FS；Industrial为21.941540%FS，差于v6 feedback best48的17.862103%FS。因此选择v5来自已查看的外部开发结果，不能表述为源域盲选。

v11的完整产物初步显示，正常BN的端点覆盖较v6锚点和对应v9正常BN续训参照改善两个外域；固定BN端点臂没有取得相同收益。下表为源验证所选best，单位NMAE %FS，越低越好。v11独立数值与执行核对245849项全部通过，本轮随后完成运行见证并启动；源尾部的具体代价随完整v11报告保留。

| 既有配方 | 源1506 | Industrial1395 | RF100151 |
| --- | ---: | ---: | ---: |
| v6 feedback best48 | 0.063613 | 17.862103 | 10.938932 |
| v9正常BN参照 best1 | 0.066743 | 18.219190 | 10.188789 |
| v9固定BN参照 best1 | 0.063696 | 18.709661 | 9.473928 |
| v11端点正常BN best1 | 0.068488 | 17.353598 | 9.643770 |
| v11端点固定BN best4 | 0.061328 | 18.916323 | 9.915663 |

本轮只比较正常BN的local模型，回答端点数据干预的效果能否迁移到另一种既有参考归一化。两臂同时改善并非预设结论。v9控制从v6初始化，不能作为v5的匹配控制，因此新增一个相同预算的v5原源控制臂。v10的ROI配方和MixStyle不加入本轮，旧4轮及65轮实验不延长。

## 锚点和唯一干预

完整锚点为`runs/geoprr_v2_tail/iteration_v5_reference_relation/readers/joint_s20262020/best.pt`，epoch57、seed20262020、原协议`geoprr_v2_reference_relation_source_only_v1`。保存的model配置保持原四字段：`bins=256`、`decoder_dim=64`、`token_dim=16`、`coupling=joint`；原生`ReferenceRelationReader`默认参考归一化为local，不添加新的显式model字段。全部496个state严格加载，参数7461060，保留全部训练权重，优化器重新初始化。

| 训练臂 | 源数据 | 训练协议 | BN策略 |
| --- | --- | --- | --- |
| `local_control` | 原`rendering/compositional_source_v1` | `geoprr_v2_generalization_source_only_v1` | 正常更新统计和affine |
| `local_endpoint` | 已完成`rendering/generalization_endpoints_v11_20261023` | `geoprr_v2_endpoint_generalization_source_only_v1` | 正常更新统计和affine |

两臂共享网络、原生全质量读数和三项loss。唯一数据差异为原1024组生成计划中前128组的512张quartile图，精确替换为相同场景／render seed／C4顺序的真实状态`[0, 0.005, 0.02, 1]`图。0指归一化量程起点，未必是绝对物理读数0；旧图与原manifest保留。其余18078行和源验证逐项相同，训练边按冻结的旧到新ID映射；不将新训练合同写成与旧合同相同。

复用[端点数据合同](GEOPRR_V2_ENDPOINT_GENERALIZATION_PROTOCOL_CN.md)和[源阶段完成记录](data/geoprr_v2_generalization_20261023/v11_source_completed.json)。当前CPU合同检查已确认，v5与v6原train、validation及exposure_edges一致，现有端点精确替换核验适用于v5合同。本轮不重跑源渲染和GT数学检查，不增加背景多样性或端点比例。

## 冻结预算与选择规则

两臂各4个完整epoch，seed20262020、fresh AdamW、shared LR1e-5、head LR5e-5、min_lr1e-6、warmup0、weight_decay0.01、gradient_clip5、AMP。train batch32、validation batch16、8 workers、768／192输入与输出。原三项精确`roleNLL/(3*log(256)) + fieldL1 + CDF`，系数各1。无新增模块、参数、ROI增强、MixStyle、固定BN或目标域训练。

每轮18590 unique源图、22686 donor曝光、8192 recipient曝光、30878训练CNN图像、709批、1506验证图。每臂共2836批、123512张训练CNN曝光、6024张验证图，两臂训练总预算247024张CNN曝光和12048张验证图；有效update、AMP skip、运行时间及峰值按实际记录，不以计划批数替代有效更新数。已经发生的v11渲染成本单独保留，本轮新增渲染为0。

只按本臂4轮内完整源1506 NMAE选择best-source，另保存last4；原best57作为初始化参照，不自动加入本轮best选择。分别评估完整Industrial1395／52组和RF100151／35组的原生ROI。best若为epoch4，则复用同一权重外评，不重复GPU。全分母保留输入失败，按error1计；原768／192与ImageNet归一化、原生读数保持不变，无目标域阈值选择、TTA或GT几何输入。RF100标量由四点离线派生，照实披露。

比较local57、对应新原源控制和端点臂的每域连续误差、Acc、零位／近零／非零、组、尾部／最大误差，及源普通1495／1497、严重6、重点3172／2810／3663／9323的逐图误差和救回／新增严重错误。v11只作为不同归一化的辅助证据，不替代本轮匹配控制。无增量则保留负结果并结束这组配方，不延长4轮或追加模块。

## 执行与归档

独立入口为`experiments.run_geoprr_v2_generalization --study local_endpoints`，输出`runs/geoprr_generalization/iteration_v12_local_endpoints_20261023/matched_training`。新队列记录两臂各自源路径、协议、共同锚点、配置、版本、PID／锁及代码快照；原v9／v10／v11协议与产物不覆盖。

正式运行前要求v11模型队列、端点源队列完整结束，旧controller与GPU任务均已退出，并完成v11一次独立数值／执行核对。随后对这两份精确配置做必要的新锚点运行见证：完整v5权重严格加载；原源合同一致、端点合同512行精确替换；实际端点RGB的loss／梯度有限；正常BN策略正确；验证输入identity；外评Reader按原生local配置严格加载。该见证优化更新为0，不是性能结果，不重复旧预检。结果保存`diagnostics/geoprr_generalization_v12_local_endpoint_preflight_20261023/result.json`，要求`status=passed`、精确两份configs、`counts.optimizer_updates=0`、`validation_input_exact_identity=true`，当前不预先生成通过产物。

两臂串行训练和外评，期间不修改live模型／配置／名单／标签，不启动额外GPU。完整阶段后收齐config、run、history、best、latest逐图、completed及外域完整逐图与成本，仅做一次必要CPU数值和执行核对。公开显式allowlist中的代码、协议、规范化标量／逐图与计数；RGB、权重、raw run／data_contract、H／C／XYZ、像素和private trace只留private。

## 限制

这是单seed、短训练的外域开发筛选，Industrial和RF100均已用于问题诊断与选方向，不能称新盲测、统计显著性或新算法认证。数据替换只增加原场景内的状态覆盖；v5与v6历史训练轨迹不同，本轮能检验local内匹配效果，不能单独归因所有归一化差异。新图／新ID保持曝光数量和边结构，但不声称逐步像素及随机光度实现完全相同。源精度与尾部退化需同时报告，不能仅凭某个外域均值选择最终模型。原14494／1506／4000、旧名单、G1／anchor失败、配对守恒负结果与全部模型保留，不扩seed或重跑SyncG officialtest。

本轮实际运行见证已通过：2次CNN调用／4张训练图像／2次反传／0优化更新，原源与端点源均保持原生local归一化、完整496 state严格加载及有限梯度，正常BN的228 buffers按预期更新。原源合同等于v5锚点原合同，端点源只作声明的512行替换；2个验证样本输入严格一致。见[运行见证](data/geoprr_v2_generalization_20261023/v12_runtime_preflight.json)和[启动记录](data/geoprr_v2_generalization_20261023/v12_training_launch.json)。
