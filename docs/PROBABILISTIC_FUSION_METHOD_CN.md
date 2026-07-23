# 概率方向估计、透视感知进度校准与选择性路由

本文新增路径保持原生产 API 和已签名的 v1 实验不变，单独实现并审计一个论文候选模型。它的
目标不是继续堆叠数据集，而是针对两个已观察到的失败模式：模糊/透视使指针方向不稳定，分割
分支与方向分支在不同样本上各自占优。

## 方法定义

### 概率圆周方向头

ResNet-18 编码器后同时预测：

- `64×64` 支点热图；
- 直接二维方向向量；
- 72-bin 圆周角度分布；
- 每张图的对数角方差。

解码时将归一化直接向量与圆周分布的 resultant vector 相加后再归一化。训练角误差使用圆周
差值，异方差项为：

```text
L_nll = 0.5 * (Δθ² * exp(-s) + s)
```

其中 `s` 是预测对数方差。它与软圆周分类损失、余弦方向损失和支点热图损失联合优化。相比
只回归一个角度，该头能够表达“预测在哪个方向”和“这个方向有多可靠”，并避免 0°/360°
边界不连续。

相对同编码器的 v1 支点—方向模型，v2 只有 18,761 个额外可训练参数（14,080,588 vs
14,061,827，约 +0.13%）；因此对比主要检验训练目标和概率表示，而不是换用更大的主干。

### 透视配对与等变一致性

每个 SyncG train crop 生成两个视图：第一个做较轻的光度增强，第二个再叠加强光度退化和
最高 45° 的虚拟 yaw/pitch 投影。支点和指针射线使用同一个单应矩阵精确变换，不使用近似
伪标签。

除两个视图各自的监督损失外，还把第一视图预测的支点和射线通过单应矩阵变换到第二视图，
与第二视图预测计算可微一致性损失。该约束是“等变”而不是强行令图像方向不变：投影后指针
在像平面中的方向本来就应发生变化。

需要注意，准确的像平面方向不能完全消除强透视下“角度到刻度进度”的非线性。因此论文要
分别报告方向 MAE 与端到端 NMAE，不能用前者代替最终读数结论。

### 不确定性软融合

原掩码分支和概率方向分支共同成功时，一个小型 MLP 预测两者的归一化读数误差对数方差
`s_mask` 与 `s_vector`，再用逆方差权重融合：

```text
w_mask = softmax([-s_mask, -s_vector])[0]
y_hat = w_mask * y_mask + (1 - w_mask) * y_vector
```

如果某一分支无输出，仍执行确定性的硬失败保护：mask 失败时使用 vector，vector 失败时保留
mask，两者都失败才返回失败。MLP 特征只来自推理时可见的分支分歧、mask 质量、支点热图、
圆周熵、resultant length 和预测方差；单测明确检查特征提取不读取真值、样本 ID、数据集名或
环境标签。默认 53 维输入、64 维隐藏层的方差网络只有 5,602 个参数。

冻结实验显示，连续软融合只适合作为消融：它在训练 OOF 上有小幅收益，但多数冻结条件不如
离散选择路由。最终方法保留该负结果，不把 inverse-variance averaging 作为主模型。

### 透视感知的角度—进度校准

方向 v2 将 severe perspective 的方向 MAE 从 3.603° 降到 1.089°，但原始端到端 NMAE
没有同比下降。原因是最终读数仍使用冻结前端的 `start_angle/range_angle`：参考角偏差会通过
`progress=(pointer_angle-start_angle)/range_angle` 放大。为此新增 progress residual calibrator：

```text
r_progress = progress_gt - progress_vector
progress_cal = clip(progress_vector + clip(r_hat, -c, c), 0, 1)
```

输入只有运行时可见的周期角度、参考分支、量程角、表盘框比例、支点质量和概率方向不确定性；
不使用样本 ID、数据集名、环境标签或真值。ExtraTrees 只是一种轻量实现，不是创新点；方法贡献
是把概率方向误差显式转化为可校准的归一化进度残差。校正上限 `c=0.30` 只由 SyncG/train
grouped-OOF 选择。

最终路由在 mask 与 calibrated vector 共同成功时预测后者的收益，并保留不可覆盖的硬失败规则。
训练侧 nested-OOF 达到 NMAE `0.08985`、Acc@2% `0.46370`；hard fallback 为
`0.11548/0.41370`。七条件冻结结果的代表值如下：

| 条件 | 旧质量路由 NMAE | 校准 vector | 最终校准路由 | 最终 Acc@2% |
|---|---:|---:|---:|---:|
| Clean | 0.10709 | 0.09562 | **0.08901** | 0.48175 |
| Severe blur | 0.12603 | 0.11304 | **0.10815** | 0.40875 |
| Severe perspective | 0.16219 | **0.14515** | 0.14539 | 0.18475 |
| Severe combined | 0.23160 | **0.21780** | 0.21837 | 0.17075 |
| RPM-10K | 0.32409 | 0.28521 | **0.28420** | 0.05509 |

六个 SyncG 条件相对旧质量路由的 group-bootstrap 区间均完全低于 0。RPM 只有六个仪表组，
点估计改善但区间跨 0，不能声称统计显著。

## 无泄漏训练协议

1. 三个方向模型只使用 SyncG train，并按 `meter type + scene` 分组留出验证集。
2. 融合训练样本只取三个验证集的并集；每行 vector 预测必须来自从未见过该仪表组的模型。
3. mask 侧使用既有 grouped-OOF 预测，而不是训练集内拟合值。
4. 融合器内部再做五折 GroupKFold，报告 nested-OOF 指标后才在全部训练 OOF 行上拟合最终模型。
5. SyncG test、六类控制退化和 RPM-10K 只用于冻结评测，不能回调权重、温度或阈值。

默认训练与评测入口：

```powershell
# 三个完整方向训练种子；每个训练结束后立即做源码/数据/逐轮/权重审计。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_probabilistic_direction_training.ps1

# 主种子：clean、两档模糊、两档透视、联合重退化、RPM-10K。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_probabilistic_direction_evaluations.ps1

# 三种子 clean/RPM 稳定性。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\evaluate_probabilistic_direction_replicates.ps1

# 训练侧跨模型 OOF、融合训练、特征消融、七条件冻结评测与逐行验证。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_uncertainty_fusion_experiments.ps1

# 进度校准、mask/calibrated-vector 路由、七条件评测与逐行复核。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_calibrated_progress_experiments.ps1
```

## 必需对比和消融

- 同任务外部基线：`VDN architecture, retrained on SyncG`；
- 内部强基线：mask、v1 独立方向头、硬失败回退、v1 质量路由；
- 最终候选：概率方向头与不确定性软融合；
- 上界诊断：逐样本 mask/vector oracle，只作分析；
- 解码消融：同一冻结权重的 direct-only、circular-only 和 fused decoder；
- 训练消融：去掉等变一致性、去掉整个 projective pair；
- 融合消融：去掉七个原生不确定性特征、只保留原生不确定性特征、硬回退。

所有端到端表同时报告 NMAE、Acc@2% 和 coverage。单样本归一化误差为
`|prediction-ground_truth| / |scale_end-scale_start|`；无输出在 NMAE 中按 1.0 计，在
Acc@2% 中算错。`NMAE=0.10` 表示平均误差相当于量程的 10%，不等于“准确率 90%”；
`Acc@2%=0.40` 表示 40% 样本的误差不超过各自量程的 2%。

## 创新性边界

圆周分类、异方差回归、单应增强、ExtraTrees 和 inverse-variance fusion 分别都不是首次提出。
论文可主张的是面向指针仪表的组合：精确投影标签下的支点—射线等变学习、圆周分布与角方差
联合解码、角度到归一化进度的无泄漏残差校准，以及 mask/calibrated-vector 安全选择路由。
是否形成有效论文贡献由冻结端到端结果、配对区间和消融决定，不能仅凭模块名称声称 SOTA。
