# 读数集合与保护普通预测：六篇主文献对照

检索时间2026-10-21。检索包含2026年近期几何不确定性工作；下表只保留6篇主文献。问题锚点是减少严重误读，同时保护普通样本和所有新增阈值错误；本阶段没有据文献启动训练。

| 原始论文 | 与拟议方法的实际重合 | 仍未解决的本任务差别 |
|---|---|---|
| [Object Pose Estimation with Statistical Guarantees，CVPR 2023](https://arxiv.org/html/2303.12246)，§4–5 | 把关键点热图转成联合预测集合，经重投影与正深度约束形成PURSE，再采样和计算最坏误差外界。 | 使用已知3D关键点及相机内参；不是未知标定跨度、逐高度材料和身份不确定的直接读数算子。不能宣称集合传播是新贡献。 |
| [Uncertainty Quantification of Set-Membership Estimation in Control and Perception，L4DC 2024](https://proceedings.mlr.press/v242/tang24a/tang24a.pdf)，§3 | 对非凸半代数集合用约束剪枝、广义松弛Chebyshev中心和SOS构造外逼近；允许选择参数子向量。 | 只挑一个标量任务坐标并不新。可能的差别必须是本任务消元带来的计算或紧致性收益。 |
| [Uncertainty Quantification for Visual Object Pose Estimation: S-Lemma Ellipsoidal Bounds，TRO 2026](https://arxiv.org/html/2511.21666)，§IV–VI，2026-08修订版 | 像素无穷范数界与正深度化为反投影约束，再经S-lemma／SOS求联合位姿椭球及角度／平移外界。 | 仍以已知内参和3D关键点为输入；不同关键点置信度需要正确组合。不能把单点边际覆盖直接当联合覆盖。 |
| [Learning Safe Prediction for Semi-Supervised Regression，AAAI 2017](https://ojs.aaai.org/index.php/AAAI/article/download/10856/10715) | 将监督预测投影到其他回归器形成的凸可行集；在真值属于该集合的条件下保护原预测。 | 区间clip是该安全投影原则的简单一维对应。实质难点是RGB证据能否形成可靠且有用的读数外界。 |
| [From Conformal Predictions to Confidence Regions，arXiv 2024](https://arxiv.org/html/2405.18601)，§2–3 | 将输出区间拉回参数集合，讨论交集／投票和带弃权回归。 | 噪声自由目标的覆盖依赖额外假设；整数截断标签不能自动满足所需噪声条件。 |
| [EPro-PnP，CVPR 2022](https://arxiv.org/html/2203.13254v4)，§3 | 从加权重投影正规化出位姿后验，并学习对应与权重，保留多种解释。 | 多假设、概率几何和对应学习均已有。需要具体的相对进度消元／外界算子，而非换成表针任务后重新命名。 |

当前结论是继续做有限、可证伪的算子验证；**不将通用可行集传播、概率拟合或保护性裁剪作为新算法贡献**。未发现上述任一论文已经包含完整GeoPRR目标，也没有证据认证尚未实现的算子具有独立创新。

最小区分性验证须使用同一RGB证据，对比单解拟合、通用几何集合求解和任务专用算子，分别检查集合包含、宽度、完整成本、普通1495／1497保护、旧严重6张及新增>1/2/5错误。全1506上的经验包含不等于覆盖证明；原冻结数值门槛保持不变。
