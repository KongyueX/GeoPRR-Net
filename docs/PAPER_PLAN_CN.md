# Robust Pointer Meter Reading 论文方案（概率方向—进度校准版）

> 截至 2026-07-23，六组控制退化、RPM 外测、VDN 公平重训、概率方向消融、进度校准和
> 最终选择路由均已完成。最终方法在 clean / severe perspective / severe combined / RPM
> 上的 NMAE 为 `0.0890 / 0.1454 / 0.2184 / 0.2842`，同协议重训 VDN 为
> `0.1482 / 0.1712 / 0.2364 / 0.3813`。七个条件的 NMAE 配对区间均低于 0，但 severe
> perspective 与 RPM 的 Acc@2% 仍略低于 VDN。RPM 只有六个表盘组；凡是与旧质量路由的
> 区间跨 0 的改善只能写作点估计。三个概率方向种子的验证角度 MAE 为
> `0.7787° ± 0.1279°`。精确数值见 [`FORMAL_RESULTS_CN.md`](FORMAL_RESULTS_CN.md)。

## 一句话问题定义

现有指针表读数流水线在清晰正视图上可以工作，但模糊、倾斜拍摄、分割噪声和合成到真实的
域偏移会同时造成错误读数与无输出。本文研究：**如何联合精确但易失败的指针掩码表示与覆盖率
更高的支点—方向向量表示，并用可审计的选择性路由在退化条件下兼顾精度、覆盖率与负迁移。**

## 建议题目

中文：

> 面向模糊与透视退化的概率方向—进度校准双表示指针表读数

英文：

> Perspective-Equivariant Probabilistic Direction and Progress-Calibrated Routing for Robust Pointer Meter Reading

标题不要写 “state-of-the-art”。RPM-10K 冻结外测只能支持“跨域诊断”，主标题使用可复现的
模糊/透视问题定义更稳妥。摘要应明确：最终方法在 RPM 和严重组合退化的 NMAE 上优于
VDN、coverage 达到同等水平，但 Acc@2% 仍较低；不能笼统声称所有指标全面优于现有方法。

## 可作为论文贡献的部分

### 1. 概率方向—进度校准双表示路由

- 掩码分支由 SyncG 微调分割、双几何融合、归一化残差和选择门控组成，精度较高但会受
  分割缺失与中心线校验影响。
- 方向分支共享一个 ResNet-18 编码器，同时预测支点热图、直接二维方向、72-bin 圆周分布
  和样本相关角方差；连续向量与周期分布联合解码。
- 训练时严格投影支点和射线标签，并约束原图/单应视图预测等变。去掉投影配对后，severe
  perspective / combined 的角度 MAE 从 `1.089° / 1.624°` 恶化到
  `1.826° / 2.630°`，这是“大视角优势”的核心消融证据。
- 原始方向角先通过仅用 SyncG train grouped-OOF 拟合的 angle-to-progress 残差校准器，
  再映射为量程读数；这一步解决“角度更准但刻度进度仍有投影偏差”的问题。
- 最终路由保留硬失败规则；两分支都成功时，才根据运行时可得的分支分歧、不确定性、支点/
  掩码质量和参考几何预测切换收益。4,380 张、197 组训练样本零组泄漏，测试标签不参与拟合。
- RPM coverage 为 `0.9878`；最终方法在七条件 NMAE 上均显著优于同协议重训 VDN，但
  severe perspective 与 RPM 的 Acc@2% 不领先。

最终决策可写为：

```text
p_cal = clip(p_raw + clip(delta_hat, -0.30, 0.30), 0, 1)
y_final = y_cal_vector, if mask fails and vector succeeds
        = failure,      if both fail
        = y_cal_vector, if both succeed and predicted_gain > tau
        = y_mask,       otherwise
```

`delta_hat`、`predicted_gain` 和 `tau` 都只从 SyncG train grouped-OOF 得到。方向分支仍共享
冻结表盘框、起终参考和已知量程，因此只能称为“分割无关方向分支”，不能称为完全无前端的
端到端模型。正文消融应包含 `mask / raw vector / calibrated vector / hard route /
quality route-v1 / final route / oracle`、投影配对与等变损失，以及 direct/circular/fused
解码。当前测试集参与过支持性诊断，严格 confirmatory 版本仍需另留现场测试集。

### 2. 双几何估计与质量感知融合

- Geometry-v1：拟合指针主轴并从表盘中心选择最远针尖。
- Geometry-v2：在外侧候选点上进行稳健方向投票，降低单像素毛刺和高光的影响。
- 每个估计器输出可解释质量量：轴线一致性、候选支持度、方向投票集中度、正反侧分离度。
- 融合权重来自上述运行时质量，不使用真值、表型标签或测试集统计。

需要通过 “Quality-weighted Fusion vs Mean Fusion” 的配对误差及分组 bootstrap 区间验证。若正式结果没有显著改善，应把它写成稳健融合组件，而不是单独声称性能提升。

### 3. 跨量程归一化残差

学习目标为：

```text
r = (y - y_geometry) / (scale_end - scale_start)
```

不同量程的表可以共享同一个残差模型。运行时残差重新乘以量程，并裁剪回合法读数区间。该设计比直接回归绝对读数误差更适合混合量程训练。

### 4. 不确定性与学习门控的选择性校正

- ExtraTrees 的树间方差作为残差不确定性。
- 门控输入只包含运行时特征：两几何分支分歧/置信度、mask 轴线统计、分割概率统计、起终点分支和椭圆代理。
- mask 面积类特征按裁剪面积归一化、距离类特征按短边归一化，避免固定像素尺度在不同分辨率上产生伪域偏移。
- 门控学习“应用残差是否比保留几何基线更好”，不直接学习测试标签。
- 残差过于不确定、门控拒绝或模型缺失时，系统回退到质量加权几何结果。

论文重点应放在“减少负迁移”，而不只是平均误差。至少报告：

- 修正覆盖率；
- 负迁移率；
- Ours-mask vs Residual without Gate；
- risk–coverage 曲线；
- 门控 Brier、ECE 和 AUROC。

### 5. 无泄漏的嵌套分组交叉拟合

外层测试组既不进入残差模型，也不进入残差裁剪分位数、门控模型或门控标签构造。外层训练部分再次进行内层分组 OOF，每个内层残差模型的裁剪值也只由对应内层训练组确定，再用无泄漏残差构造门控标签。最终部署模型的裁剪值使用完整 SyncG train，门控阈值只从其外层 OOF 预测确定。

这是可信实验设计的一部分，也可写成方法贡献；不要把普通随机 K-fold 描述成同等方案。

### 6. 成对控制退化与端到端失败计分

对同一批 SyncG test 图像施加由 `seed + sample_id` 固定的高斯模糊和 25°/45° 虚拟平面
透视，所有方法共享完全相同的退化输入和冻结校准器。除困难条件下的绝对方法差值外，额外
报告相对 clean 的成对 ΔΔNMAE；无输出按 NMAE=1、Acc@2%=失败计入并单报 coverage。
这是一项评测协议贡献，而不是新网络模块。正式结果支持“相对原始 Transformer，在困难
条件下仍保留绝对优势和风险控制”，不支持“相对退化幅度总是更小”，也不支持在所有困难
条件下优于 VDN。

## 支撑性改进，不宜包装为核心创新

- 修复 U2NetP Letterbox 输出恢复时未移除补边的坐标错误。
- 只使用 SyncG train 微调 U2NetP，采用加权 BCE + Dice 深监督。
- 损失、阈值校准和分割指标都只统计去除 Letterbox 补边后的真实内容区。
- 从 SyncG train 内按 `gauge_type::scene_name` 分组留出验证集。
- 在该验证集上、经过最大连通域后处理后冻结概率阈值。
- released 与 fine-tuned 权重都在同一验证划分上校准阈值，组件对比不混入阈值选择差异。
- 混合精度前向配合 FP32 logit loss，并记录被 GradScaler 跳过的优化器步数。
- 表盘框轻微抖动及颜色、模糊增强用于模拟检测误差与真实成像差异。

这些内容能证明实现严谨并提升前端，但 U2NetP、YOLO、ExtraTrees、Dice loss 本身都不是本文首创。

## 不能主张的内容

- 不能把使用开源数据库写成算法创新。
- 不能把 RPM-10K 派生子集称为完整官方 2000 张榜单。
- 不能声称在 RPM-10K 上训练或微调；正式协议完全不使用其 8730 张训练图。
- 不能只在成功样本上报告准确率并忽略失败。
- 不能把仓库原有 30+8 张内部图称为两个独立公开数据集。
- 不能只凭内部消融声称优于同类型公开模型或达到 state of the art。
- 不能声称本方法在所有模糊/透视强度下相对 clean 的退化小于 Transformer。
- 不能声称本方法在重度透视、组合退化或 RPM 的所有指标上全面优于 VDN；最终校准路由的
  NMAE 更低，但这些条件的 Acc@2% 仍不都领先。
- 不能把独立方向分支描述为 VDN 改进版；二者训练协议可比，但本文实现不依赖 VDN 源码。
- 不能声称最终路由在每个条件都优于 calibrated vector；重度透视和组合退化下后者分别低
  `0.00024 / 0.00057`。
- 不能声称 RPM 相对旧质量路由的改善显著；该比较只有六组且区间跨 0。
- 不能声称每个质量特征都是必要创新；训练侧消融显示单个分支分歧已能取得大部分收益。

## 两数据集的最小实验协议

### SyncG

- 官方 train：分割微调、分组验证、残差/门控拟合和所有阈值选择。
- 官方 test：一次冻结同域测试。
- train/test 不互换；test 不用于早停或选择分割阈值。
- 正式入口同时核对 16000/4000 个官方样本 ID 的固定哈希，而不只核对文件数量。
- 固定 Hugging Face 提交的数据卡许可为 CC BY 4.0，论文和衍生模型说明中保留来源归属。

### RPM-10K single-pointer subset

- 只使用官方 test 标签和图像。
- 固定保留六种主表型中、标量读数位于 `[0, range]` 的 1797 张。
- 固定排除：194 张 `others`、7 张非标量读数、2 张越界标签。
- 选择规则只读取标签结构，不读取任何模型输出。
- 官方 `range` 作为已知量程元数据，因此结果不等同于从原始图像同时推断量程的官方 VLM 设置。
- 数据许可当前为 TBD，仓库不得再分发图像。

## 外部对比主表与内部消融必须分开

论文主对比保留五行同任务方法。VDN 的公平重训与七组冻结评测已经完成；HARR 官方权重也已
完成 Pointer-10K pointer-branch 零样本评测，但因训练来源不同只进入跨协议补充表。其他公开
工作因缺少可用权重/代码或协议不同，不填入推测成绩。可复现性状态与公平适配规则见
[`BASELINE_AUDIT_CN.md`](BASELINE_AUDIT_CN.md)。

| 主对比方法 | 公平设置 |
|---|---|
| Classical geometry (Geometry-v1) | 相同表盘框、起终点与已知量程，不使用学习残差 |
| Original Transformer | 原项目同任务模型 |
| VDN (our retraining) | 在相同 SyncG train 重训，共享量程适配器；同时报告角度误差与读数指标 |
| Ours-mask | 质量融合 + 归一化残差 + 不确定性/学习门控 |
| Ours-final | 概率方向 + 透视感知进度校准 + SyncG-train-only grouped-OOF 安全切换 |

其中 `Geometry-v1` 可作为传统几何实现；若篇幅有限，正文保留 Classical geometry、
Original Transformer、VDN retraining、Ours-mask、Ours-final 五行。Ours-hard、原七个读数
变体以及 `vector only / oracle` 属于内部消融，不能冒充外部方法。

每个数据集至少报告端到端 NMAE、Acc@阈值和 coverage。失败读数在 NMAE 中按 `1.0` 量程误差计，在 Acc 中算错；同时保留 `successful_nmae` 作为诊断，不能作为主结论。

RPM-10K 额外报告：

- 官方公式的成功输出 Ref 与 Rel（不截断），并同时报告 coverage；
- 全样本分母的 Accε（Ref≤1%）与 Accθ（Rel<5%，零真值除外），无输出直接算错；
- 10/100 失败惩罚的截断误差只作额外稳健性诊断，不称为官方榜单指标；
- 六表型及环境条件分层结果。

## 最小补充实验

正文只需要：

1. 五行外部主对比：传统几何、Original Transformer、VDN 重训、Ours-mask、Ours-final；
2. 七方法原分支消融，以及 `mask only / vector only / hard-failure dual route`；
3. clean、两级模糊、两级透视、严重组合退化的鲁棒性表与退化曲线；
4. 分割组件消融：released vs SyncG-finetuned（同一 train-validation 校准阈值；真值表盘框，明确标注非端到端）；
5. RPM-10K 上 released vs SyncG-finetuned 分割的冻结迁移诊断，禁止据此选前端；
6. risk–coverage 曲线、负迁移率和关键配对的分组 bootstrap 95% 区间；
7. 从同一冻结缓存统计的端到端失败归因表，不额外运行模型；
8. 三随机种子的残差/门控稳定性汇总；视觉预测保持冻结，只重复该学习模块；
9. 独立支点—方向头的三次完整训练，并在 clean 与 RPM 上汇总均值 ± 样本标准差。
10. 概率方向三种子、direct/circular/fused 解码、投影配对/等变损失消融；
11. angle-to-progress 校准和最终 grouped-OOF 路由的七条件冻结复评；
12. 与同协议重训 VDN 的独立外部主表，以及 HARR 官方权重的跨协议 Pointer-10K 附表；
13. batch=1 全链路平均/中位/P95 延迟及按失败阶段分解；其他论文只进入相关工作表。

附录可加入：

- geometry-only features；
- no-mask/segmentation-confidence；
- no-ellipse；
- disagreement-only。

这些消融只复用预测缓存并重拟合小型树模型，不新增数据集，也不重复昂贵视觉推理。

## 结果判定与止损规则

- 如果 SyncG test 上 Ours-mask 不优于 Quality-weighted Fusion：检查残差裁剪、组划分和门控覆盖率；不改 RPM 参数。
- 如果 Residual without Gate 改善但 Ours-mask 无改善：门控目标或阈值过保守，应只用 SyncG train OOF 诊断。
- 如果 RPM coverage 很低：首先检查表盘检测/指针分割失败占比；不能只汇报成功子集 NMAE。
- 如果 SyncG 微调分割降低 RPM 性能：如实报告合成域微调的负迁移，保留 released 前端作为预声明对照，不能根据 RPM 标签反向选择阈值。
- 当前进度校准路由已解决大部分“有输出但低质量”问题。不得根据七组冻结结果继续改特征或阈值；
  若再改模型，必须另设未使用的最终测试集，并把当前结果降级为开发集诊断。

## 复现入口

```powershell
.\experiments\download_syncg.ps1 -Extract
python -m experiments.download_rpm10k_test
.\experiments\run_paper_experiments.ps1
.\experiments\run_robustness_experiments.ps1
.\experiments\run_seed_stability.ps1
.\experiments\run_vdn_evaluations.ps1
.\experiments\run_pivot_direction_experiments.ps1
.\experiments\run_pivot_direction_replicates.ps1
python -m experiments.summarize_pivot_direction_replicates
python -m experiments.collect_quality_router_oof --overwrite
python -m experiments.train_quality_router --overwrite
python -m experiments.ablate_quality_router_features --overwrite
python -m experiments.summarize_quality_router --overwrite
python -m experiments.verify_quality_router_run
.\experiments\run_probabilistic_direction_training.ps1
.\experiments\run_probabilistic_direction_evaluations.ps1
.\experiments\run_probabilistic_direction_ablations.ps1
.\experiments\run_calibrated_progress_experiments.ps1
```

正式数值只允许从 `artifacts/runs/` 自动生成；文档和论文中不得手工填写未经脚本复核的结果。

## 2026-07-23 技术路线更新

方向分支已由单一单位向量升级为“直接向量 + 72-bin 圆周分布 + 可学习角度方差”的概率方向头，
并用严格成对的单应投影视图和等变损失训练。三种子验证角度 MAE 为
`0.7787° ± 0.1279°`；在 severe perspective / severe combined 上，冻结角度 MAE 从旧版的
`3.603° / 4.997°` 降到 `1.089° / 1.624°`。

更重要的是，新增的透视感知进度校准器把方向角改进传递到了最终读数。它只使用 SyncG train
grouped-OOF 样本学习归一化 angle-to-progress 残差，再由第二层安全路由在 base mask 与
calibrated vector 之间选择。最终 clean / severe blur / severe perspective / severe combined /
RPM NMAE 分别为 `0.0890 / 0.1082 / 0.1454 / 0.2184 / 0.2842`；旧质量路由对应为
`0.1071 / 0.1260 / 0.1622 / 0.2316 / 0.3241`。

正文方法主线因此应更新为：

```text
概率圆周方向估计
-> 精确投影配对监督与等变正则
-> 透视感知的 angle-to-progress 残差校准
-> mask/calibrated-vector 防泄漏选择路由
-> 全分母失败计分与分组 bootstrap
```

论文中仍需明确：树模型本身不是创新；严重透视和严重组合下 calibrated vector 单路由略优于
最终路由；RPM 改善的 95% CI 跨 0；当前测试集已参与支持性诊断，严格确认性投稿应另留现场测试集。
