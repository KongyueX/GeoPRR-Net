# R²MT-Net 架构与命名边界

## 1. 唯一论文主线

论文中的主模型只有 **R²MT-Net**（Representation-Conditioned Multi-Risk
Moment Transport Network）。完整任务是从现场图像得到物理读数：定位仪表、
估计归一化指针进度、识别量程端点，再完成单位换算。

```text
现场图像
  ├─ 定位与裁剪 ─> ROI ─> R²MT-Net ─> 归一化进度 p
  └──────────────────────> OCR ─────> 有序量程端点 (s_min, s_max)

最终读数 = s_min + p × (s_max - s_min)
```

OCR 是端到端系统不可缺少的接口，但不是本文的主要算法创新。因此实验仅呈现
通过量程有效性检查的已接受样本精度，不以 OCR 覆盖率或实时延迟支撑
R²MT-Net 的核心结论。

## 2. 为什么单一风险修正不够

投影畸变不是一个同质误差源。实验中的三类目标会强调不同样本：

- mean-risk 关注总体平均误差；
- tail-risk 关注少量大误差；
- combined-risk 额外强调严重透视与模糊叠加条件。

单个头无法同时达到三种目标的最优。三种 specialist 均弱于多风险组合；固定
三风险先验已经取得主要收益，样本级 router 只带来小幅增益。因此本文把
“多风险互补”作为主要证据，把 representation-conditioned routing 解释为保守
细化，避免夸大 router 的独立贡献。

## 3. 四项机制各自解决的问题

### 3.1 共享双观察编码

原始 ROI 与 support-normalized ROI 由同一套 ResNet-18 参数分别前向。这样既保留
原始纹理，又引入近似正视观察，同时不增加第二个图像编码器。标量端点经固定
尺度 posterior lift 转为 128-bin 分布，保证 posterior 的一阶矩严格等于端点预测。

### 3.2 显式多尺度关系传输

仅依赖归一化图像会丢失原始观察信息，简单拼接又不能表达两种观察之间的几何
对应。内部 relation-transport foundation 对齐 stride-8/16 特征，并编码有符号差、
绝对差、乘积一致性、公共 support 和 homography 几何量，再用 progress-bin query
解码为一个 base posterior。该 foundation 是 R²MT-Net 的内部组成，不作为独立
论文模型；旧代码中的 `ReMST` 仅为该内部模块和历史权重的兼容标识。

### 3.3 多风险矩残差与保守路由

三个 RCMT 头具有相同结构、不同训练风险。router 读取深层 raw/SARN 表征关系、
几何量、端点统计以及三个候选 shift，并围绕固定先验
`(0.475, 0.280, 0.245)` 产生受限调整。论文设置的 adaptive strength 为 0.5，
目的是利用样本差异，同时避免不稳定的完全自适应专家选择。

### 3.4 单次精确 posterior transport

R²MT-Net 先在矩空间组合三个 residual，再对完整 base posterior 做一次 exponential
tilt，使最终一阶矩等于目标值。消融显示，先分别传输三个 posterior 再做概率混合
会显著恶化误差；因此“先仲裁矩、后单次传输”是必要的融合顺序。当关系证据无效
时，shift 被置零，最终 posterior 与 raw posterior 完全一致。

## 4. 训练与推理边界

- 主评估单位为归一化满量程误差（NMAE, %FS）。
- 三个独立拟合种子用于报告均值与样本标准差。
- relation foundation、三个风险头和 router 分阶段拟合；最终推理冻结全部参数。
- 公开入口为 `r2mt.load_r2mt_net`；内部 checkpoint protocol 保留旧名称以兼容已完成
  的拟合结果，不应出现在论文图表的模型对比中。
- `R²MT-Net` 与 `ReMST` 不是两条并列方法线。后者最多在消融表脚注中说明内部
  checkpoint 对应关系。

## 5. 证据解释

当前 SyncG、工业 ROI、VDN intersection、OCR accepted-output、identity replay 与效率
结果均为 retrospective evidence。它们支持“面向投影畸变的多风险矩传输”这一窄化
结论，但不能替代独立多站点外部验证。若论文进一步声称自动系统覆盖率或端到端
实时性，则还需使用 live detector、R²MT-Net、OCR 和单位换算的完整重放。
