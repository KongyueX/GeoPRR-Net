# v13固定CNN权重、更新源BN：完整结果

2026-10-23。**固定CNN权重使两个原生家族的源均值更接近任务锚点，但Industrial与RF100的均值均差于各自完整微调端点控制；这一配方不保留为跨域改进，不延长训练。** feedback best1的源／Industrial／RF100为0.064050／18.384623／10.710380%FS；local best4为0.077136／21.581871／9.174795%FS。相对匹配完整微调best，源均值分别下降6.4797%／7.5020%，两个外域却均回退，源尾部也没有全部解决。

固定的是7,203,152个CNN参数，源BN运行统计仍更新，decoder和head的257,908个参数继续训练；模型总量仍为7,461,060。这不是固定所有CNN状态，更不是恢复ImageNet预训练表征。所有权重、负结果与逐图记录保留。

## 同人口best与固定末轮结果

误差单位为NMAE %FS，越低越好。best-source仅由新四轮完整源1506选出；末轮固定第4轮。local best4与last4为同一checkpoint，其源表和外评复用，不重复推理。

| 模型／配方 | checkpoint | epoch | 源1506 | Industrial1395 | RF100151 |
|---|---|---:|---:|---:|---:|
| v6_feedback48 | anchor | 48 | 0.063613 | 17.862103 | 10.938932 |
| v11_endpoint | best_source | 1 | 0.068488 | 17.353598 | 9.643770 |
| v11_endpoint | last | 4 | 0.081321 | 18.007754 | 10.329537 |
| v5_local57 | anchor | 57 | 0.077123 | 21.941540 | 9.159439 |
| v12_local_endpoint | best_source | 1 | 0.083392 | 20.539125 | 7.659157 |
| v12_local_endpoint | last | 4 | 0.090847 | 19.624347 | 7.835575 |
| v13_feedback_cnnfixed | best_source | 1 | 0.064050 | 18.384623 | 10.710380 |
| v13_feedback_cnnfixed | last | 4 | 0.066119 | 18.679481 | 10.851050 |
| v13_local_cnnfixed | best_source | 4 | 0.077136 | 21.581871 | 9.174795 |
| v13_local_cnnfixed | last | 4 | 0.077136 | 21.581871 | 9.174795 |

匹配完整微调控制来自[v11 feedback端点](../v11_completed/report_cn.md)和[v12 local端点](../v12_completed/report_cn.md)，其源／外部逐图直接链接复用，未重算或复制。两个家族训练历史不同，横向差异作为上下文，不视为孤立归一化效应。

## 外域连续效果、准确率与组

Δ为本轮减去同家族完整微调控制，单位为NMAE百分点。best与best、last与last分别匹配；正值表示误差上升。

| 本轮臂／checkpoint | 数据集 | ΔNMAE百分点 | 相对控制误差上升% | 逐图改善／恶化 |
|---|---|---:|---:|---|
| feedback_cnnfixed best_source | industrial | +1.031026 | 5.9413 | 621/774 |
| feedback_cnnfixed best_source | rf100 | +1.066611 | 11.0601 | 57/94 |
| feedback_cnnfixed last | industrial | +0.671727 | 3.7302 | 623/772 |
| feedback_cnnfixed last | rf100 | +0.521513 | 5.0488 | 66/85 |
| local_cnnfixed best_source | industrial | +1.042746 | 5.0769 | 656/739 |
| local_cnnfixed best_source | rf100 | +1.515638 | 19.7886 | 56/95 |
| local_cnnfixed last | industrial | +1.957524 | 9.9750 | 520/875 |
| local_cnnfixed last | rf100 | +1.339220 | 17.0915 | 56/95 |

| 臂／checkpoint | 数据集 | Acc@1／2／5%FS（%） | 最差10%均值／max %FS |
|---|---|---|---|
| feedback_cnnfixed best_source | industrial | 7.956989/13.118280/26.523297 | 56.269856/88.315421 |
| feedback_cnnfixed last | industrial | 8.100358/13.189964/27.025090 | 56.963028/89.982557 |
| local_cnnfixed best_source | industrial | 9.247312/14.336918/26.021505 | 61.884468/89.588189 |
| feedback_cnnfixed best_source | rf100 | 33.112583/53.642384/67.549669 | 49.737301/65.338755 |
| feedback_cnnfixed last | rf100 | 32.450331/52.317881/65.562914 | 49.354923/63.931489 |
| local_cnnfixed best_source | rf100 | 30.463576/52.980132/67.549669 | 43.626742/62.495703 |

全量、每个组／component、零位子集、连续逐图、救回／新增与尾部分位数均保存在[完整指标](metrics.json)。本轮并非每个样本均退步，但两个家族在best与last两种比较中，两个数据集的总体均值都差于自己的完整微调控制。不能用源验证改善替代跨域结论。

## 源普通人口、严重错误与重点样本

| 臂／checkpoint | ordinary1495 | ordinary1497 | 原严重6均值 | >1／>2／>5数 | 最大误差及ID |
|---|---:|---:|---:|---|---|
| feedback_cnnfixed best_source | 0.058333 | 0.058689 | 1.298723 | 4/2/1 | 5.041761，sync_3663 |
| feedback_cnnfixed last | 0.058416 | 0.058719 | 1.653568 | 5/2/1 | 7.182678，sync_3663 |
| local_cnnfixed best_source | 0.066215 | 0.070580 | 1.310935 | 5/4/1 | 6.604299，sync_3172 |
| local_cnnfixed last | 0.066215 | 0.070580 | 1.310935 | 5/4/1 | 6.604299，sync_3172 |

| 臂／checkpoint | ordinary1495 median／p90 | 源全量最差10%均值 |
|---|---|---:|
| feedback_cnnfixed best_source | 0.033568/0.122705 | 0.304793 |
| feedback_cnnfixed last | 0.033021/0.123353 | 0.323789 |
| local_cnnfixed best_source | 0.043982/0.143850 | 0.350651 |
| local_cnnfixed last | 0.043982/0.143850 | 0.350651 |

相对v11 feedback完整微调best1，feedback冻结best1在>1%FS救回6906、无新增；>2%FS无救回或新增；>5%FS救回2810、但新增3663，故源max虽由9.506392降至5.041761%FS，严重尾部仍在。feedback冻结last4相对其完整微调last4救回2810的>5%FS，但3663仍为7.182678%FS。

相对v12 local完整微调best1，local冻结best4的3172由18.677585降至6.604299%FS，但新增338的>1／>2%FS；>5%FS无成员增减，3172仍是原有严重错误。相对完整微调last4，冻结local救回12540的>5%FS，同时新增338的>2%FS。完整参考模型逐项比较及ID均保留，不将源均值改善称为严重错误全部修复。

| 模型／checkpoint | 3172 | 2810 | 3663 | 9323 |
|---|---:|---:|---:|---:|
| feedback完整微调 best_source | 0.002092 | 9.506392 | 4.470917 | 0.102615 |
| feedback完整微调 last | 0.983444 | 7.003999 | 25.798727 | 0.084943 |
| local完整微调 best_source | 18.677585 | 0.005883 | 0.020537 | 0.112969 |
| local完整微调 last | 21.480571 | 0.011414 | 0.372794 | 0.105739 |
| feedback_cnnfixed best_source | 0.116268 | 3.632623 | 5.041761 | 0.046504 |
| feedback_cnnfixed last | 0.114757 | 3.276002 | 7.182678 | 0.065953 |
| local_cnnfixed best_source | 6.604299 | 0.001526 | 0.207822 | 0.106263 |
| local_cnnfixed last | 6.604299 | 0.001526 | 0.207822 | 0.106263 |

## 零位、非零与近零

| 臂／checkpoint | Industrial零588／非零807 | RF100零6／非零145 | Industrial正近零99 | RF100正近零19 |
|---|---|---|---:|---:|
| feedback_cnnfixed best_source | 23.943754/14.334104 | 0.409345/11.136630 | 23.544597 | 23.388206 |
| feedback_cnnfixed last | 24.045979/14.769318 | 0.505226/11.279153 | 23.481980 | 23.304054 |
| local_cnnfixed best_source | 27.828869/17.030154 | 12.692097/9.029252 | 31.291771 | 12.812267 |
| local_cnnfixed last | 27.828869/17.030154 | 12.692097/9.029252 | 31.291771 | 12.812267 |

零位按原生FP32 target==0、非零按target!=0，正近零为0<target≤0.05。两个外域均无0.95≤target≤1样本，完整指标保留样本数0，不给出不存在的满量程泛化成绩。冻结CNN并未保留完整微调的RF100零位收益，尤其local零位由完整微调best1的2.343347升至12.692097%FS；该子集仅6张。

## 权重／BN策略、输入与真实成本

分别严格加载完整原生feedback48或local57，496 state、7,461,060总参数；fresh AdamW、seed20262020、各4轮。CNN的7,203,152个参数固定，228个参数tensor在每臂4／4轮保存检查中均未变；decoder228096与head29812参数训练，CNN可训练参数0，合计可训练257908。源BN正常更新：每轮检查228个buffers，其中152个running mean／variance发生变化，76个batch counters均增加709。固定参数与更新BN是同一明确策略，不称CNN所有state不变。

使用与各自v11／v12完整微调控制直接相等的端点数据合同，512替换、18078保留及原验证不变的证据沿用已完成记录；不重算几何或GT。每轮18590 unique、22686 donor＋8192 recipient、30878训练CNN图像／709批、源验证1506，原三项loss各1、shared LR1e-5／head5e-5／min1e-6、warmup0、train32／val16、8 workers、768／192，无ROI增强或MixStyle。

| 臂 | 训练批 | 训练／验证CNN图像* | 有效更新／AMP skip | 训练／验证wall秒 | 训练process秒 | CUDA峰值allocated／reserved bytes |
|---|---:|---|---|---|---:|---|
| feedback_cnnfixed | 2836 | 123512/6024 | 2836/0 | 794.767/32.451 | 837.107 | 5622561792/12538871808 |
| local_cnnfixed | 2836 | 123512/6024 | 2836/0 | 799.093/31.513 | 842.295 | 5646081024/13161725952 |

*训练图像数来自保存曝光与拼接forward实现，不作为独立instrumented调用计数；每臂90744 donor＋32768 recipient、123512训练图像与6024验证。无resume、两臂均2836次有效更新，AMP skip0。相比对应完整微调，记录的process时间与峰值allocated较低，但这项成本权衡没有转化为更好的两域误差，不构成独立速度基准。

| 臂 | 实际外评份数 | CNN图像／调用（记录值） | forward／pipeline／process秒 | backward／update |
|---|---:|---|---|---|
| feedback_cnnfixed | 4 | 3092/196 | 35.702/45.770/57.240 | 0/0 |
| local_cnnfixed | 2 | 1546/98 | 17.818/22.713/28.407 | 0/0 |

实际6份外评、4638张CNN图像／294次调用，0反传／0更新；local best4=last4的外评复用，别名计新增0。源逻辑best／last保存共6024行，公开只保存3份不同表／4518条表行，local last通过[别名记录](checkpoint_aliases.json)指向best4，避免重复表。全部声明输入与预测有效，完整1395／151分母和失败error1规则保持不变，RF100标量沿用四点离线派生，推理只用ROI RGB。

本轮复用已有真实端点RGB，新增渲染为0；历史物理源生成成本保留在[源阶段记录](../v11_source_completed.json)，不重复计入v13训练或外评。此前运行见证与正式成本分开，保存执行证据用于记录，无新增模型检查。

完成证据的一次独立CPU数值与执行核对267479项全部通过、失败0；数值最大差1.42e-14，三项loss汇总算术残差7.22e-10低于1e-8。固定权重／BN证据来自保存runtime metadata；未重新加载checkpoint、执行GPU、CNN、反向或优化器，也未重复渲染GT、教师或旧核对。见[数值执行核对](numeric_execution_audit.json)与[公开清单](public_allowlist.json)。

## 后续

结束CNN权重冻结这组配方，保留源精度／资源代价权衡及完整负结果，不追加轮次。独立v14已完成任务锚点与本家族完整微调端点source-best1的固定alpha=0.5权重插值及完整评估：源均值与最大误差有所恢复，但两个外域均值均差于完整微调参照。没有追加训练、alpha扫描或目标域调参，见[完整中点结果](../v14_completed/report_cn.md)与[锚点插值协议](../../../GEOPRR_V2_ANCHOR_INTERPOLATION_PROTOCOL_CN.md)。

## Limitations

单seed、各4轮，Industrial和RF100均为已查看的外域开发人口；不称新盲测、显著性、普遍机制或新算法证明。source1506仅在本轮新epoch中选best，last4单列；local best／last同权重且外评复用。家族先前训练历史不同，横向差异不能孤立归因结构。CNN参数固定但源BN buffers更新，不保持全部CNN状态，也不等同ImageNet表征保留。端点合同与固定权重依据保存metadata及已有独立核对；没有新的checkpoint加载或GT验证。RF100标量来自四点离线标注、零位只有6张。原14494／1506／4000、普通1495／1497／严重6名单、旧G1／anchor失败与配对守恒负结果保留；RGB、权重、raw run／合同、H／C／XYZ和像素只私有保存。
