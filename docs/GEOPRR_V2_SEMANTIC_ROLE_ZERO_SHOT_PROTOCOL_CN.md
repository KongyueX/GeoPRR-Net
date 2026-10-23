# 完整语义直接条件角色能量：独立zero-shot协议

2026-10-23。v21按本文件冻结一个独立源域结构与60轮预算；实际模型构造与6项新CPU检查已通过，8823492参数／503 state／263参数tensor已核实；源RGB零更新见证及启动尚待完成，没有新性能或启动结论。代码SHA在本轮实现提交后记录。Industrial1395与RF100151不得用于训练、适配或选择模型／训练超参数；全部旧模型、原G1／anchor失败、65轮与配对守恒负结果保留。

## 可检验问题

[v20完整结果](data/geoprr_v2_semantic_relation_20261023/completed/report_cn.md)已归档并通过唯一54062项CPU数值／执行核对：源均值优于v16／v18，两个外域均退化。代码中的完整semantic1280显式通路只经4个校准输出影响角色，没有独立的semantic→role energy通路；FPN仍有全局感受野。这是一个可检验的结构问题，不是已证明的外域退化原因。

本轮保留v20的局部几何与联合读出，只增加每图semantic直接调节角色能量的通路，检验它是否能在源域学到有效的角色条件，并改善冻结模型的跨数据集读数。不重新运行v20，不恢复训练好的checkpoint，不把本轮描述为延长旧60轮或已成立的新算法。

[Conditional DETR（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_Conditional_DETR_for_Fast_Training_Convergence_ICCV_2021_paper.html)区分定位空间条件与内容匹配，通过decoder embedding学习条件空间query，约束定位范围并减轻对content embedding的依赖，提供将定位与内容条件分别检验的一般动机。本轮新增的是完整图像语义产生的角色内容query，不复现其DETR decoder或空间query机制；三角色与16维设计来自现有v20接口，文献不证明本任务zero-shot收益或退化原因。

## 模型与初始化

先完整构造fresh v20模型：已有缓存的完整`mobilenetv4_conv_medium.e500_r224_in1k` encoder，保留conv_head／norm_head和实际1280维semantic；共享四尺度FPN64、semantic校准context、径向角色采样、local参考条件能量与原CircularProgressPushforward均保留。四尺度为blocks.0／1／2／4，通道48／80／160／960，stride4／8／16／32。完整旧502项初始state构造完成后，才附加一个无bias的Linear1280→48，weight全零；复用原模块与buffer，不另初始化或复制训练好的任务权重。

新矩阵产生每图3个16维query，分别对应A／P／B，维数沿用现有token_dim16。令s为完整semantic、z_r(θ)为原角色投影token，唯一能量变化为：

`q = reshape(W*s, [3,16])`，

`unary'_r(θ) = unary_r(θ) + dot(q_r, z_r(θ))/sqrt(16)`。

新增项在原角度角色能量中计算，再沿用原local参考条件／相对索引。缩放为0.25，只有61440个新参数与1个state，不增加bias，因为原角色unary已存在。query维数、缩放及无bias是一次固定设计，不搜索。原A–P／A–B双线性能量、共同A条件、local归一化、相对角差／跨度联合J及全质量运输都不变，不能将J替换为两个边缘分布的乘积。

`RGB → 一次完整encoder → semantic校准context + semantic角色query + 多尺度空间tokens → 原local条件联合J → 原全质量运输 → reading`。

query全零时，原502项state和初始reading必须与同seed fresh v20精确一致，须由实际构造／运行记录验证后才标为已通过。实际CPU构造核实完整8823492参数／503 state／263个参数tensor，非CNN388980参数／32个tensor。所有CNN、FPN、校准、原角色与新增query参数均参加源训练。初始W=0时，首先学习的是W的梯度；不能声称新query已经向semantic encoder贡献非零梯度，encoder仍通过原有校准／空间路径学习。

模型训练、验证和外域均只接收RGB并执行同一部署函数。每图query是在固定权重下读取该图RGB的条件计算，不是目标域适配或总体统计拟合。BN只由源训练更新，eval固定；没有新scalar point、融合MLP、router、C4轨道、CDF、confidence门控或GT推理输入。只用已有ImageNet缓存fresh初始化，HF_HUB_OFFLINE=1，不下载新foundation。

## 源合同与唯一loss

沿用v20的18590 unique源训练与原1506验证：原14494图及4096组内状态，复用v11已完成的512张真实端点替换，原图、标签与manifest不改，不新增渲染、配对重复曝光或seed。端点0／1是归一化量程起点／终点，0不必是绝对物理读数零。256 RGB、OpenCV LINEAR、ImageNet归一化、原contrast／saturation 0.8–1.2和brightness ±0.1保持，无新增ROI扰动或MixStyle。

H／span、64网格解析field U和ring／arc mask M、虚拟A／P／B角色只用于源loss。P是半径1的虚拟dial-plane状态点，不是实体针尖。沿用原有向C4／ordered ticks／物理校准限制，不剪裁虚拟P、不改端点或删除失败行。原14494／1506／4000与ordinary1495／1497、severe6及历史重点名单保持。

唯一loss固定为：

`SmoothL1(deployed_reading,y,beta=0.05) + 0.1 * (role_NLL/(3*log(256)) + field_L1)`。

与v20保持同一loss及权重，不增加CDF或其他监督。复用v18全源解析合同记录和已完成的[理想角色联合映射兼容性](data/geoprr_v2_semantic_relation_20261023/source_role_mapping_feasibility.json)，不重新执行旧全源检查／Gaussian Q／教师／GT审计。该理想标签记录只说明源离散标签到原读出的兼容性，不证明可学习性能或外域效果，也不构成本轮新Q数学保证。outside_arc_mass只作连续诊断；理想端点读数1可同时具有约0.999997的弧外质量，不能硬门控正确性。

## 固定训练、选择与成本

独立60完整epoch，fresh AdamW、weight_decay1e-4、gradient_clip5、seed20262020、AMP、train64／val64／workers4。LR同v20／v16：epoch1–30为`3e-4*(1+cos(pi*(e-1)/30))/2`；31–60为`3e-5*(1+cos(pi*(e-31)/30))/2`。不扩seed、精度策略、query维数或其他参数搜索，不触碰officialtest4000。

每轮完整源1506验证，只按部署reading NMAE选择最早严格最小best，last60仅记录源结果。全部60结束后，才对冻结source-best各一次Industrial1395／52组、RF100151／35组。目标不得用于训练loss、总体特征统计拟合、BN适配、checkpoint／早停／epoch、阈值、ROI、温度或训练超参数选择。推理只有原生ROI RGB；RF100 scalar由四点离线派生，仅作计分。输入或scalar失败均在完整原图分母中保留normalized error1，不过滤坏预测。

计划每轮18590训练图／291调用、1506验证图／24调用；正式训练1115400图／17460次反传尝试，验证90360图／1440调用，外域1546图／25调用，总1207306图／18925次CNN。query／relation／readout调用分别为训练17460、验证1440、目标25，每ROI仍一次CNN，目标反传和更新0。实际updates＋AMP skips须为17460，额外query计算、各阶段时间、process／full、峰值内存和resume分别实记；相同CNN曝光不是相同计算或推理成本，不能报计划为已完成。

## 必要新检查与执行

计划模块`experiments.geoprr_v2_semantic_role`，入口`experiments.run_geoprr_v2_semantic_role`，独立输出`runs/geoprr_generalization/iteration_v21_semantic_role_conditioning_20261023/source_only_training`。新入口已实现并通过6项针对性CPU检查（0.194秒），尚未做真实源RGB运行见证或启动训练；代码SHA由正式提交确定。

新结构CPU检查只针对真实变化：零query时能量／J／reading与fresh v20一致；共同A和相对gather、partial batch保持；固定空间tokens及校准时，非零semantic query能实际改变角色能量；新增query梯度、RGB-only输入与失败分母符合声明。实际参数／state计数须核实，不重复旧GT／G0／教师／完整审计或评分评审循环。

一次实际源RGB零更新见证计划为4次CNN／10图呈现／6 unique图／1次combined backward／0optimizer：fresh v20 eval两张源0／1图，fresh零query v21 eval同两图，新模型train四张g0／g1端点并反传，最后eval两张原生验证图。旧base占1次／2图，新模型占3次／8图。必须核实502项初始state精确相同及1项全零query、实际1280输入和B×3×16 query、新W梯度有限且非零、旧encoder／FPN／context／角色路径梯度有限，263个参数tensor无optimizer保持不变。77个BN counter只在新源train各+1，eval与base eval不更新；完整503 state严格reload，GT不进入forward，两个验证256输入identity。正式训练fresh重新初始化，不沿用见证的BN状态；见证不是性能结果。

启动前必须v20 complete且controller／子进程退出，实际GPU空闲，配置／代码／协议／预算冻结，新CPU及实际源见证通过。唯一队列依次source60→Industrial→RF100，不在阶段交接GPU短暂为空时重复启动。运行以pipeline PID、queue current_step、原日志、history／latest和实际GPU命令判断；best时间不变不是停滞，DataLoader workers不是额外模型。活job期间只监控，不改live code／model／config／名单／标签或新增GPU评估。

保存原protocol／snapshot、全部503 state与optimizer／scaler／RNG、history、源best／latest和各1506逐图、两套完整外域逐图／组／summary及真实成本。真实实现故障先查看当前step日志及活命令，保留partial与失败，在同output从latest恢复；实质方法／配置／人口变化另立独立实验，不删行或绕保护。

全阶段完成后仅一次必要独立CPU数值与执行核对，再比较v20／v16／v18及原几何和旧GeoPRR保存参照的各域NMAE／Acc、zero／nonzero／nearzero、组、ordinary1495／1497、severe6、3172／2810／3663／9323／8195、sync9／4025、RF41／6／46／140、tail／max／救回新增及成本。只公开本任务代码和显式规范化allowlist，RGB、权重、H／C／XYZ／U／M、bbox、特征矩阵、raw run／data_contract与private trace保留私有；只推现有main，不建branch。

## Limitations

零初始化与相同base state只控制初始模型，不认证全部后续随机步骤完全相同。语义条件双线性角色匹配是常见结构手段，本轮不作创新认证或单因素泛化因果声明。v20混合结果仅提出问题；新query可能无效或负迁移，理想源标签兼容性也不保证模型学习正确。两外域已查看、只有单seed，不称新盲测、SD、显著性或普适机制；旧GeoPRR像素／完整权重链和原生输入训练史不同，未重新认证。全部旧失败保留，无收益不自动升最终模型，不延长60轮、叠模块或追加seed。
