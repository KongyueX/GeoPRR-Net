# Robust Pointer Meter Reading 论文方案（冻结版）

> 2026-07-21 的正式实验与六组控制退化实验已经完成。精确数值、分组 bootstrap 区间、
> 失败归因和消融结论见
> [`FORMAL_RESULTS_CN.md`](FORMAL_RESULTS_CN.md)。正式结果表明残差校正
> 是主要读数增益来源；学习门控改善 Acc@2% 并降低负迁移，但不改善
> clean 平均 NMAE，因此门控应定位为风险控制模块，而不是精度主贡献。在模糊和
> 大视角下，本方法保持显著的绝对 NMAE 优势，但相对 clean 的退化不小于 Transformer。

## 一句话问题定义

现有指针表读数流水线在清晰正视图上可以工作，但模糊、倾斜拍摄、分割噪声、两种针尖估计分歧和合成到真实的域偏移会造成不可预测的大误差。本文研究的不是重新堆叠一个更大的视觉模型，而是：**如何利用运行时几何质量与不确定性，在模糊和大视角条件下修正读数，并在校正可能产生负迁移时主动回退。**

## 建议题目

中文：

> 面向模糊与大视角退化的质量感知几何融合与选择性残差校正

英文：

> Quality-Aware Geometry Fusion and Selective Residual Calibration for Pointer Meter Reading under Blur and Perspective Distortion

标题不要写 “state-of-the-art”。RPM-10K 冻结外测只能支持“跨域诊断”，主标题使用可复现的
模糊/透视问题定义更稳妥；在 VDN 等外部同类基线真正完成以前，摘要也不要声称领先现有方法。

## 可作为论文贡献的部分

### 1. 双几何估计与质量感知融合

- Geometry-v1：拟合指针主轴并从表盘中心选择最远针尖。
- Geometry-v2：在外侧候选点上进行稳健方向投票，降低单像素毛刺和高光的影响。
- 每个估计器输出可解释质量量：轴线一致性、候选支持度、方向投票集中度、正反侧分离度。
- 融合权重来自上述运行时质量，不使用真值、表型标签或测试集统计。

需要通过 “Quality-weighted Fusion vs Mean Fusion” 的配对误差及分组 bootstrap 区间验证。若正式结果没有显著改善，应把它写成稳健融合组件，而不是单独声称性能提升。

### 2. 跨量程归一化残差

学习目标为：

```text
r = (y - y_geometry) / (scale_end - scale_start)
```

不同量程的表可以共享同一个残差模型。运行时残差重新乘以量程，并裁剪回合法读数区间。该设计比直接回归绝对读数误差更适合混合量程训练。

### 3. 不确定性与学习门控的选择性校正

- ExtraTrees 的树间方差作为残差不确定性。
- 门控输入只包含运行时特征：两几何分支分歧/置信度、mask 轴线统计、分割概率统计、起终点分支和椭圆代理。
- mask 面积类特征按裁剪面积归一化、距离类特征按短边归一化，避免固定像素尺度在不同分辨率上产生伪域偏移。
- 门控学习“应用残差是否比保留几何基线更好”，不直接学习测试标签。
- 残差过于不确定、门控拒绝或模型缺失时，系统回退到质量加权几何结果。

论文重点应放在“减少负迁移”，而不只是平均误差。至少报告：

- 修正覆盖率；
- 负迁移率；
- Ours vs Residual without Gate；
- risk–coverage 曲线；
- 门控 Brier、ECE 和 AUROC。

### 4. 无泄漏的嵌套分组交叉拟合

外层测试组既不进入残差模型，也不进入残差裁剪分位数、门控模型或门控标签构造。外层训练部分再次进行内层分组 OOF，每个内层残差模型的裁剪值也只由对应内层训练组确定，再用无泄漏残差构造门控标签。最终部署模型的裁剪值使用完整 SyncG train，门控阈值只从其外层 OOF 预测确定。

这是可信实验设计的一部分，也可写成方法贡献；不要把普通随机 K-fold 描述成同等方案。

### 5. 成对控制退化与端到端失败计分

对同一批 SyncG test 图像施加由 `seed + sample_id` 固定的高斯模糊和 25°/45° 虚拟平面
透视，所有方法共享完全相同的退化输入和冻结校准器。除困难条件下的绝对方法差值外，额外
报告相对 clean 的成对 ΔΔNMAE；无输出按 NMAE=1、Acc@2%=失败计入并单报 coverage。
这是一项评测协议贡献，而不是新网络模块。正式结果支持“困难条件下仍保留绝对优势和风险
控制”，不支持“相对退化幅度总是更小”。

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

论文主对比至少保留四类同任务方法；没有 VDN 等外部复现结果时，不应投稿时声称完整的
同类模型比较。可复现性状态与公平适配规则见
[`BASELINE_AUDIT_CN.md`](BASELINE_AUDIT_CN.md)。

| 主对比方法 | 公平设置 |
|---|---|
| Classical geometry (Geometry-v1) | 相同表盘框、起终点与已知量程，不使用学习残差 |
| Original Transformer | 原项目同任务模型 |
| VDN (our retraining) | 在相同 SyncG train 重训，共享量程适配器；同时报告角度误差与读数指标 |
| Ours | 质量融合 + 归一化残差 + 不确定性/学习门控 |

其中 `Geometry-v1` 可作为传统几何实现；若篇幅有限，可只保留 Classical geometry、
Original Transformer、VDN retraining、Ours 四行。下面七个内部变体属于消融表，不能冒充
外部方法：Original Transformer、Geometry-v1、Geometry-v2、Mean Fusion、
Quality-weighted Fusion、Residual without Gate、Ours。

每个数据集至少报告端到端 NMAE、Acc@阈值和 coverage。失败读数在 NMAE 中按 `1.0` 量程误差计，在 Acc 中算错；同时保留 `successful_nmae` 作为诊断，不能作为主结论。

RPM-10K 额外报告：

- 官方公式的成功输出 Ref 与 Rel（不截断），并同时报告 coverage；
- 全样本分母的 Accε（Ref≤1%）与 Accθ（Rel<5%，零真值除外），无输出直接算错；
- 10/100 失败惩罚的截断误差只作额外稳健性诊断，不称为官方榜单指标；
- 六表型及环境条件分层结果。

## 最小补充实验

正文只需要：

1. 四行外部主对比：传统几何、Original Transformer、VDN 重训、Ours；
2. 七方法内部消融表；
3. clean、两级模糊、两级透视、严重组合退化的鲁棒性表与退化曲线；
4. 分割组件消融：released vs SyncG-finetuned（同一 train-validation 校准阈值；真值表盘框，明确标注非端到端）；
5. RPM-10K 上 released vs SyncG-finetuned 分割的冻结迁移诊断，禁止据此选前端；
6. risk–coverage 曲线、负迁移率和关键配对的分组 bootstrap 95% 区间；
7. 从同一冻结缓存统计的端到端失败归因表，不额外运行模型。

附录可加入：

- geometry-only features；
- no-mask/segmentation-confidence；
- no-ellipse；
- disagreement-only。

这些消融只复用预测缓存并重拟合小型树模型，不新增数据集，也不重复昂贵视觉推理。

## 结果判定与止损规则

- 如果 SyncG test 上 Ours 不优于 Quality-weighted Fusion：检查残差裁剪、组划分和门控覆盖率；不改 RPM 参数。
- 如果 Residual without Gate 改善但 Ours 无改善：门控目标或阈值过保守，应只用 SyncG train OOF 诊断。
- 如果 RPM coverage 很低：首先检查表盘检测/指针分割失败占比；不能只汇报成功子集 NMAE。
- 如果 SyncG 微调分割降低 RPM 性能：如实报告合成域微调的负迁移，保留 released 前端作为预声明对照，不能根据 RPM 标签反向选择阈值。
- 只有当中心无关方向 fallback 在 SyncG train-validation 上满足预先固定的线性度、方向唯一性和负迁移约束时才加入；当前 10 张 RPM 诊断不足以授权该算法进入主表。

## 复现入口

```powershell
.\experiments\download_syncg.ps1 -Extract
python -m experiments.download_rpm10k_test
.\experiments\run_paper_experiments.ps1
.\experiments\run_robustness_experiments.ps1
```

正式数值只允许从 `artifacts/runs/` 自动生成；文档和论文中不得手工填写未经脚本复核的结果。
