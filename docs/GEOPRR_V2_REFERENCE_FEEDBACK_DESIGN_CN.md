# 角色兼容性能量反馈参考：设计与匹配对照

本候选只改变参考分布A的归一化，使pointer／end的内容兼容性能量参与参考权重。MobileNetV4／FPN、四参数角度线束、径向token、角色匹配、标签和全质量读数保持原合同。完整模型仍为7,461,060参数，无新loss、温度、行中心化或硬门槛。结构与两臂配置已经实现；执行合同见[正式协议](GEOPRR_V2_REFERENCE_FEEDBACK_PROTOCOL_CN.md)。正式阶段状态与产物按该协议归档。

## 完整1506诊断依据

固定v5 best57的一次完整前向已完成，原读数重放最大差1.20×10⁻⁷；所有干预保留1506人口与全部质量。以下单位均为%FS，teacher结果只作条件诊断。

| 固定读出／干预 | 全1506 NMAE | 最大误差 | 共同普通1495 mean |
|---|---:|---:|---:|
| 原best57 native joint | 0.077123 | 5.908847 | 0.067571 |
| 只换teacher中心／SPD重采样 | 0.074296 | 6.069162 | 0.065187 |
| teacher坐标下替换A | 0.064343 | 0.816369 | 0.063658 |
| native缓存只替换A，不重采样 | 0.066918 | 0.959477 | 0.066338 |
| native对A硬取MAP | 0.121517 | 2.823281 | 0.118431 |

sync_3172原误差5.908847；只换teacher坐标后为6.069162；在teacher坐标下替换A降到0.074193，而单换P、B及P+B仍分别为6.010446、6.053087、5.994335。teacher-A使完整1506全部≤1%，但它依赖GT；native缓存替换A的结果也只说明权重干预敏感性。硬取MAP虽然降低最大误差，却损害普通样本精度，因此候选保留软参考分布。

现有A的前向值由起点token的点式unary产生，不读取P／B兼容性logits；其训练梯度仍可经联合读数、CDF与共享特征耦合到其他模块。这里检验的是增加显式前向反馈是否减少错误参考权重，不能据GT干预宣称已证明唯一训练根因。

## 唯一结构改动

对固定图像token，保留现有三个投影与unary。绝对索引a、p、b均为256个圆周bin，D=16：

`eP(a,p)=qA(a)ᵀqP(p)/√D+uP(p)`，`eB(a,b)=qA(a)ᵀqB(b)/√D+uB(b)`。

保留原条件分布`P(p|a)=softmax_p eP(a,p)`、`B(b|a)=softmax_b eB(a,b)`。在行softmax之前计算：

`zP(a)=logsumexp_p eP(a,p)`，`zB(a)=logsumexp_b eB(a,b)`，

`A_feedback(a)=softmax_a[uA(a)+zP(a)+zB(a)]`。

不能从已归一化的log_P／log_B求z；那些行的logsumexp为0。两个z是来自角色兼容性的反向消息，不增添可学习参数。

保留循环相对索引`P_rel(a,d)=P(a+d|a)`、`B_rel(a,s)=B(a+s|a)`，以及

`J(d,s)=Σ_a A_feedback(a) P_rel(a,d) B_rel(a,s)`。

完整三角色分布等价于`p(a,p,b)=exp[uA(a)+eP(a,p)+eB(a,b)]/ZG`，其中`log ZG=logsumexp_a[uA(a)+zP(a)+zB(a)]`。不需显式存储256³张量。

## NLL与读数

对一个角色组合，原表达式`−log A(a)−log P(p|a)−log B(b|a)`变成

`log ZG−uA(a)−eP(a,p)−eB(a,b)`。

两个行partition项精确相消；对原factorized two-hot角色标签取期望后同样成立。因此保留原`role_nll/(3log256) + gauge_aligned_field_l1 + progress_cdf`，三项权重仍为1，不新增监督。条件P／B的GT绝对角色按每个参考a转换到相对索引，不能只监督argmax参考。

δ／span仍为圆周256格；span=0／full-turn alias均匀进入progress，其余按原中点展开、除span、clip与two-hot映射。保留全部质量，不使用r8、direct回退或GT选择读数。推理输入仅RGB ROI，虚拟pointer phase不等于物理针尖。

全部token共同循环移位，只会同样置换eP／eB的行列、z与A；相对J及读数不变。沿用该整数roll不变性，不新增绝对phase编码。

## 匹配删除对照与训练设计

| 固定项目 | 两臂共同取值 |
|---|---|
| 候选／删除 | feedback使用上式；local仅用`A_local=softmax_a uA(a)` |
| 其余关系层 | 两臂均为joint读出；P／B条件地图公式相同 |
| 初始化 | 完整v5 best57权重与buffer逐tensor相同，不仅加载骨干／FPN |
| 模型／seed | 7,461,060参数；唯一20262020 |
| 训练长度 | 65轮；前5轮冻结共享模块及其统计，第6轮开始共同微调 |
| 训练人口 | 18590 unique；22686 donor item＋8192 recipient／轮 |
| 实际曝光 | 30878图像前向、709执行batch／轮；有效update与AMP skip分别记录 |
| 尺寸／batch | 输入768、输出192；训练32、源验证16；8个worker |
| 优化 | 相同新建AdamW；共享lr=1e−4、头lr=5e−4、weight decay=.01、clip=5 |
| 调度／数值 | CosineAnnealingLR，T_max=65、min_lr=1e−6；AMP骨干、FP32头／loss、FP64标签极分解 |
| 标签／目标 | 原v5标签和三项loss，自第1轮权重1有效，无loss／窗口矩阵 |
| best选择 | 全1506源验证NMAE严格最小，平局保留最早 |

两臂仅更改A归一化；相同初始权重不要求初始输出相同。这是新的反馈删除实验，不补跑旧v5 product。拟先完成feedback 65轮并检查原G1；未过则归档并跳过local训练，通过后才运行同seed local 65轮。训练输出与正式停止规则须在实施协议中冻结后再启动。

## 验证门槛与判定

原14494／1506／4000名单、旧严重6、共同普通1495、r8普通1497及原11门槛保留。沿用已完成的G0标签rawmax≤1%FS，并要求10项RGB检查全部通过：

- 完整1506有效；NMAE≤0.1299504985%FS，worst151≤0.7205782515%FS，max严格<19.5571363%FS。
- 旧严重6张全部≤2%FS；相对普通监督native及固定r8均无新增>1／2／5错误。
- 共同普通1495的mean／median／P90分别≤0.0972051518／0.0564150512／0.2246701717%FS；r8普通1497 mean≤0.0977505576%FS。

同时与当前best57逐图列出全部指标、严重错误、救回／新增和普通人口保护，不能把通过旧r8门槛等同于超过best57。反馈机制筛选还要求相对匹配local的NMAE至少降低5%，worst151和>5数量不增加；同时NMAE不差于best57、相对best57无新增>2／>5严重错误，共同普通1495 mean／median／P90及r8普通1497 mean不差于best57。任何逐图择优、基于entropy的阈值路由或重选读数均不进入本候选。

若两臂差距小，即使均优于best57，也不能把继续训练收益归给反馈；若平均改善但新增严重错误，仍拒绝robust成功。报告完整执行批／有效update／AMP skip、图像曝光、训练与验证wall、实际峰值，不以preflight替代正式成本。

## 实施准入

启动前需完成最小代码修改、两臂配置和独立输出合同，并检查：全局能量与分解概率一致；two-hot NLL抵消；固定token下两臂条件P／B相同；循环roll／全质量读数；相同行条件PMF但不同row offset确实改变feedback A；best57参数／buffer完整匹配；少量实际RGB零优化前向／反传与CSV接口。不得因GT-A条件结果好看直接启动，亦不重复已完成的旧几何或完整教师诊断。

## Limitations

全局归一化、CRF和label-bias讨论已有[Andor等，ACL 2016](https://aclanthology.org/P16-1231/)等工作。该先例不证明当前全图可见的星形模型具有同一严格表达力结论，也不认证新颖性；本候选只检验任务角色兼容性对参考权重的反馈及其匹配收益。

z不等于正确性或置信度。对某一行全部eP加c(a)，条件P不变而feedback A改变；行offset与query范数可能放大错误参考，须保留为可证伪风险。全局NLL能监督这些能量，但不保证学到正确解释。P与B在给定a后仍条件独立，同一参考内的配对缺失、错误局部证据和错误标定仍可能失败。

teacher标定／角色干预含GT，重采样也有分布变化，全部属于条件诊断。新实验仅为单seed、反复查看的源开发筛选，不提供显著性、盲测或真实迁移结论；不扩seed、不运行新official test，不恢复旧专家／OOF／router矩阵。
