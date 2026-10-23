# MobileNetV4任务锚点与端点微调模型的固定中点检验

2026-10-23。状态：两个候选及全部6份完整推理已结束，一次独立CPU数值／执行核对142810项通过；见[完整结果](data/geoprr_v2_generalization_20261023/v14_completed/report_cn.md)。承接用户授权的跨数据集自主迭代，保留全部原模型、失败与负结果。

## 问题和依据

检验能否通过同一家族内的固定权重中点，保留部分端点覆盖的外域收益并减少源域退化。v11 feedback与v12 local的完整微调端点版均优于各自匹配原源控制，但源严重样本恶化。v13固定CNN权重的两臂已完成，保存外评均值均差于各自完整微调端点参照；源尾部与成本随完整核对报告保留。本轮不由这些差异推断CNN漂移的唯一因果关系。

Wortsman等的[Robust fine-tuning of zero-shot models，CVPR 2022](https://arxiv.org/abs/2109.01903)提出在原模型与微调模型之间作权重插值；[原文附录B](https://arxiv.org/html/2109.01903v3)讨论缺少额外域知识时的固定中点。这里借用可检验构造，两个左端点均为已经过仪表任务训练的模型。任务锚点不是零样本模型，本实验不等同于原论文的WiSE-FT设置。

## 固定候选和state合同

只构造以下两个候选，预先固定端点微调模型的系数为0.5：

| 候选 | 左端点：任务锚点 | 右端点：已有源验证best |
| --- | --- | --- |
| `feedback_half` | v6 feedback best48，原生global | v11正常BN `endpoint` best1 |
| `local_half` | v5 local joint best57，默认local | v12正常BN `local_endpoint` best1 |

右端点沿用各自完整四轮中按源1506选出的best。两个家族分别构造，不混合跨家族权重，不用v13作为端点，不追加last混合、alpha扫描或BN策略矩阵。原两端模型与v13的分数及逐图表均复用已有结果，不重复GPU评估。

可学习参数逐元素采用`0.5*anchor + 0.5*endpoint`，包括CNN、BN affine、lateral／refine和head。BN `running_mean`与`running_var`也按同一中点计算；这只是确定的buffer插值，不是混合后分布的精确统计估计。BN整数`num_batches_tracked`复制锚点。所有构造期固定buffer，包括bool／int索引、离散读出坐标和虚拟角色方向，必须在两端逐项相同后复制锚点。不能无区别地平均全部state。

两端原生模型配置、state key／shape／dtype严格相同；完整496 state加载，仍为7461060参数。feedback保留显式global字段，local保留原四字段和隐式local。浮点参数及BN统计须有限，方差非负。CPU针对检查验证浮点参数／BN统计中点、整数计数器复制、固定buffer不匹配拒绝及相同两端identity；不运行旧GT或教师检查。

新候选使用独立`geoprr_v2_task_anchor_half_interpolation_v1` checkpoint协议，记录alpha、两端协议／epoch／来源和state策略。其epoch、best_epoch、best_nmae为空，明确没有新的训练或best选择，不继承任何一端的optimizer／scheduler身份。文件写入独立目录，原checkpoint不覆盖。只作eval，不构造optimizer或恢复训练，不重估源或目标BN统计。

## 完整评估与成本

两个候选分别完整评估源1506、Industrial1395／52组、RF100151／35组。保持原生ROI RGB、768／192、ImageNet归一化、batch16、原生Reader和FP32读出；源端沿用原验证入口，模型只接收image，GT几何仅在原有验证准备／计分合同中使用。目标label／GT H／center／SPD／针尖不作训练或推理输入，不调ROI、阈值、温度或TTA，不使用目标BN统计。

计划共6104张CNN图像、386次CNN调用、0 backward、0 optimizer；实际调用用hook记录，时间和峰值内存按真实边界保存。每个候选推理只需要单模型一次前向；参数插值不是两模型输出集成。没有新增训练、渲染、模块、loss或参数。

实际完成6104张CNN图像／386次调用，0 backward／0 optimizer。两臂CPU准备各确认259个可学习参数tensor、152个BN moment取中点，76个BN counter复制锚点，9个构造期buffer在两端及原生模板逐项相等后复制；原生496 state严格加载。

原两端与v13逐图均复用，不新增端点评估。本轮只在alpha0.5进行新的完整推理；不根据结果改alpha或另挑checkpoint。全分母保留失败，按normalized error1计；RF100目标沿用四点离线派生标量并披露。

独立输出为`runs/geoprr_generalization/iteration_v14_anchor_half_20261023/frozen_evaluation`，入口`experiments.run_geoprr_v2_anchor_interpolation`。启动前要求v13完整核对通过、v11／v12／v13队列complete、原父子退出、实际GPU空闲，以及本轮CPU state检查通过。固定代码版本和协议后串行完成两个候选的全部评估，保留协议、组件来源、state策略、逐图、组、摘要和实际成本。

完成后只进行一次必要独立CPU数值／执行核对。分别与本家族锚点、完整微调端点版和v13比较连续误差、Acc、zero／nonzero／nearzero、组、源ordinary1495／1497、原严重6、3172／2810／3663／9323、tail／max和全部救回／新增错误。报告保留了多少端点收益及付出的源域代价，不恢复零新增错的一票否决，也不自动提升候选。

## 限制

独立CPU核对复用保存的准备和运行证据，未重新读取checkpoint tensor内容。两个外域已反复查看，属于开发结果；单seed既有模型的确定插值不构成新盲测、显著性或新算法认证。两个家族锚点训练史不同，只作家族内匹配判断。同轨迹、同架构不保证非线性读出和BN中点函数良好，也不保证严重样本随alpha平滑变化。没有新增背景多样性或解决数据覆盖全部缺口；本轮不证明恢复ImageNet表征。原14494／1506／4000、旧名单、G1／anchor失败及配对守恒负结果保留。RGB、权重、原H／C／XYZ、raw run／data_contract与private trace不上传，公开仅显式allowlist中的规范化标量与逐图结果。
