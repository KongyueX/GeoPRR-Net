# 完整1506张：逐顶点物理接口与相对状态网格

**新的物理接口同时保住普通样本精度并救回原严重误读；连续读出和冻结256×256网格均通过预注册零训练门槛。**它逐顶点使用各自高度，不再依赖origin/outside同高。下面是GT几何标签诊断。后续joint RGB模型已完成65轮并被冻结源开发门槛否决，独立组合对照按协议跳过；见[正式训练负结果](../geoprr_v2_relative_state_20261021/README.md)与[冻结训练协议](../../GEOPRR_V2_RELATIVE_STATE_PROTOCOL_CN.md)。

## 完整人口与结果

原1506张源验证全部保留，无失败、删图、按质量筛样本或GT择优混合。普通监督对照为已完成seed20262020、best63、7695954参数的同一checkpoint；native与固定r8均复用既有逐图输出。所有误差单位为%FS，Tail10为最差151张的平均误差。

| 读出 | NMAE ↓ | Acc@1 / 2 / 5，% ↑ | >1 / >2 / >5 ↓ | median / p90 / p95 ↓ | Tail10 ↓ | 最大误差 ↓ |
|---|---:|---|---|---|---:|---:|
| 普通监督native | 0.147924067 | 99.269588 / 99.667995 / 99.800797 | 11 / 5 / 3 | 0.057125 / 0.239308 / 0.389268 | 0.897472932 | 26.627396047 |
| 同模型固定r8 | 0.129950499 | 99.402390 / 99.667995 / 99.867198 | 9 / 5 / 2 | 0.057386 / 0.237127 / 0.381035 | 0.720578251 | 19.557136297 |
| GT逐顶点高度＋共同旋转＋刻度拟合 | 0.018703482 | 100 / 100 / 100 | 0 / 0 / 0 | 0.010706 / 0.045759 / 0.061047 | 0.074733253 | 0.325174493 |
| 同物理标签的冻结ordered256网格 | 0.018970809 | 100 / 100 / 100 | 0 / 0 / 0 | 0.010695 / 0.046405 / 0.062660 | 0.076500069 | 0.312203169 |

两种物理读出的最大误差均发生在`sync_10304`。连续几何相对native为1266张改善、240张变差，相对r8为1268改善、238变差。网格读出分别为1265/241及1267/239；逐图变差完整保留，没有把均值更低写成每图都更好。两种物理读出相对任一对照的新增>1、>2、>5错误均为0。

上轮[原origin/outside检查](../geoprr_v2_endpoint_geometry_20261021/README.md)仍是负结果：NMAE0.166169994%FS，最大1.053687579%FS，新增2张>1%FS，同高无标签保证。当前结果使用新的密集材料顶点接口，不能反过来证明旧两点前提成立。

## 普通样本精度保护

共同普通集合C在计算候选前已经固定：两对照误差均≤1%FS，共1495张；只用于评价分层，不用于训练采样、模型输入或路由。另报告r8自身普通1497张，避免共同集合遗漏r8已正确的图。

| 共同普通1495张 | NMAE | median | p90 | 最大误差 |
|---|---:|---:|---:|---:|
| native | 0.097923096 | 0.056415051 | 0.228086710 | 0.966954231 |
| r8 | 0.097205152 | 0.056511164 | 0.224670172 | 0.967013836 |
| GT逐顶点物理读出 | 0.018516360 | 0.010732132 | 0.045593819 | 0.325174493 |
| GT标签ordered256网格 | 0.018792022 | 0.010697544 | 0.046217442 | 0.312203169 |

r8普通1497张的mean为0.097750558%FS，连续物理读出和ordered256分别为0.018503771与0.018778613%FS。共同集合的mean/median/p90及r8普通集合mean均满足冻结非退化门槛。全体、两类普通集合的p95、Tail10和最大值均保存在[metrics.csv](metrics.csv)。

## 严重误读与新增错误

| 两对照>2%FS并集 | native | 固定r8 | GT逐顶点读出 |
|---|---:|---:|---:|
| sync_10202 | 26.627396 | 16.561826 | 0.003724 |
| sync_10635 | 2.455890 | 2.455461 | 0.003447 |
| sync_12540 | 1.364576 | 2.511204 | 0.051909 |
| sync_338 | 19.577676 | 19.557136 | 0.143262 |
| sync_3663 | 2.311166 | 3.186789 | 0.004751 |
| sync_4515 | 17.546070 | 0.904751 | 0.017438 |

连续物理读出相对native的>1、>2、>5 rescues/new分别为11/0、5/0、3/0；相对r8分别为9/0、5/0、2/0。ordered256网格的三档救回与新增名单相同。6张严重病例及两个物理读出的完整值保存在[severe_cases.csv](severe_cases.csv)，全部阈值病例保存在[threshold_cases.csv](threshold_cases.csv)；所有逐图记录保留原分母。

## 物理输入与读数合同

1. 原annotation提供按网格顺序投影的全部针体顶点，不是可见分割轮廓。三种顶点数274/76/472对应冻结scaled模板；仅以顶点数确定模板，不读y、pointer angle/rotation或已有模型误差。
2. 原integer pixel先取cell中心`+0.5`。H把原外壳框归一化坐标映射到世界z=0的canonical平面。对每个顶点分别用其模板高度与相机中心求射线交点，不要求任意两点同高。
3. 全部对应顶点经centered二维Procrustes拟合共同object rotation θ，平移残差另记。刻度序列经自由圆心/半径的复数最小二乘拟合第一刻度方向φ₀；不把新版source的刻度高度0.002当作原数据假设。
4. 相对状态为`δ=(φ₀−θ) mod 2π`和`λ=(tick_count−1)*spacing`。使用固定arc-midpoint展开、比值及[0,1]裁剪。读数不能误用带有模板针端固有侧偏的`θ+β`。
5. 先保存target-blind几何预测，再加入原float32目标和已存普通监督输出计分。失败保留error1及完整1506分母。本次1506全部有效且无clip，完整原始方向最大误差0.325174493%FS，不依靠clip通过门槛。

相机C、平面H、3D模板和原材料点是本阶段GT几何输入，新RGB模型前向不会读取它们。公开包保留读数、相对状态和标量fit诊断；不包含RGB、原始annotation、完整mesh模板、vendor或private trace。

## 冻结256×256读出可行性

δ网格为256点圆周`[0,2π)`，λ网格为256点有序`linspace(0,2π)`。δ使用circular twohot、λ使用linear twohot；从这两个几何标签形成的四个邻接atom质量，经同一物理比值推送到256个progress bin。

λ两端0和2π不表示有效有端点物理弧，其质量全部均匀分配到progress，完整记入`invalid_mass`；不删质量、不条件重归一、不回退旧模型或r8。其余弧外atom经固定展开后clip端点，另记`outside_arc_mass`。当前1506标签的invalid_mass最大0，float32分布质量误差最大1.1920929e−7。

这个标签网格检查通过与连续几何相同的全部普通保护、严重救回和新增错误门槛。它仅排除了当前离散化和读出接口的阻断，不能保证RGB预测的多峰分布也准确。

## 独立复核

连续物理接口由独立NumPy程序重做逐顶点ray lift、SVD求proper旋转及刻度复数拟合，不导入主decoder。1506张几何/指标/普通集合/全部rescues与new名单对齐，预测最大差6.66134e−16；原始目标和基线误差复现差0，见[independent_verification.json](independent_verification.json)。

ordered256另由JavaScript float64四atom推送计算复核，不导入Torch或reader。相对已归档float32读出，预测最大差2.09317e−7，独立分布质量误差最大2.22045e−16；原始读取均不使用y或基线。逐图预测误差列均重新计算，见[ordered256_independent_verification.json](ordered256_independent_verification.json)。这两个计算精度不同，差异不冒充模型不稳定。

## 全部已记录成本

| CPU阶段 | 人口 | 记录wall，秒 | 记录峰RSS，KiB |
|---|---:|---:|---:|
| 主几何decoder | 1506 | 1.003493 | 39228 |
| 独立几何与计分复核 | 1506 | 1.032339 | 187108 |
| 全部已记录标签导出，2次合计 | 每次18590＋1506 | 22.419157 | 382372，取峰值 |
| 原训练监督字段清查 | 14494 | 5.397106 | 未单独记录 |
| ordered256标签网格检查 | 1506 | 0.894225 | 未单独记录 |

最终canonical schema导出为上述标签成本的一部分，wall10.707696秒、峰RSS382184KiB，不能与合计再次相加。原annotation精简导出未保存独立wall timer，因此不以SSH耗时替代CPU计算时间。计分与公开归档没有完整端到端计时。这些边界是已记录的几何/标签CPU工作；新RGB读取、模型前向、GPU调用、训练步和seed扩展全部0，没有新增已训练模型参数。原普通监督参数为7695954。新模型参数、吞吐、进程wall和峰值须在正式run中实测，当前不填造估计。完整可读成本记录见[costs.json](costs.json)。

## 文件

- [predictions.jsonl.gz](predictions.jsonl.gz)、[predictions.csv.gz](predictions.csv.gz)：相同1506规范化逐图行，含连续/网格结果、目标、固定两基线、普通集合和fit诊断。
- [summary.json](summary.json)、[metrics.csv](metrics.csv)：完整、共同普通及r8普通人口；median/p90/p95/tail/max齐全。
- [ordered256_summary.json](ordered256_summary.json)、[ordered256_protocol.json](ordered256_protocol.json)：实际冻结网格及完整评价。
- [paired_changes.json](paired_changes.json)、[severe_cases.csv](severe_cases.csv)、[threshold_cases.csv](threshold_cases.csv)：救回、新增和逐图不利结果。
- [protocol.json](protocol.json)、[independent_verification.json](independent_verification.json)、[ordered256_independent_verification.json](ordered256_independent_verification.json)：物理合同与独立复核。
- [label_preparation_summary.json](label_preparation_summary.json)、[label_schema_verification.json](label_schema_verification.json)：原14494/新增4096/验证1506完整标签与canonical source_role合同。
- [正式RGB协议](../../GEOPRR_V2_RELATIVE_STATE_PROTOCOL_CN.md)：单seed joint→有条件independent、完整质量读出与预注册停止规则；joint完整65轮已否决，independent跳过。

## Limitations

这是GT标签几何诊断，不是自动RGB结果、性能上界、稀疏端点验证或真实迁移证据。新方法只提出“比值之前保留δ/λ配对”的待证机制，角度读数和mixture joint建模均有强近邻；没有论文新颖性认证。原H生成时使用float casing box，而原JSON只保留integer box和点，残差仍包含量化与框不一致；没有按y修正。原1506annotation没有逐图generation revision或selected vertex身份，模板对应由完整数值一致性支持。该源验证已反复查看，不称盲测；单seed不报告跨seed样本SD。
