# MobileNetV4相对物理状态分布：冻结协议

## 当前状态与待证问题

完整1506张源验证上的新物理标签接口已经通过零训练门槛：逐顶点使用各自高度反投影，再拟合共同针体旋转及有序刻度框架，不依赖origin/outside同高。连续几何读出的NMAE为0.0187034824%FS、最大误差0.325174493%FS；它保住预先固定的普通集合精度并救回原严重误读。完整逐图、负变化、独立复算与成本见[物理接口结果包](data/geoprr_v2_physical_interface_20261021/README.md)。

**joint已完整完成65轮，best epoch 60，冻结G1/G2未通过，independent已按协议跳过。**完整1506张源验证NMAE为0.4533216657%FS、最大误差82.8413169831%FS、>2/>5为62/21张；共同普通1495张均值0.3727221250%FS。结果劣于普通监督native和r8，不能作为方法收益。完整逐图、普通保护、新增错误和成本见[正式训练负结果包](data/geoprr_v2_relative_state_20261021/README.md)。服务器队列为`runs/geoprr_v2_tail/iteration_v4_relative_state`，代码版本`cd4680f`。首段的零训练接口数值来自GT几何标签；本段为RGB自动模型的源开发结果，两者不混作性能或真实迁移证据。旧origin/outside直接读数的0.166169994%FS及同高阻断仍然成立，保存在[原端点检查](data/geoprr_v2_endpoint_geometry_20261021/README.md)。

本次只验证一个机制：**在形成读数比值之前，保留相对针角δ与刻度弧长λ的配对不确定性，是否比先拆成两个独立边缘分布更能减少严重错误，同时保持普通样本精度。**低秩混合、角度分类及几何监督本身不作为独立创新点。物理和计算接口可测试，不代表论文新颖性已经通过认证。

## 人口、标签与信息边界

原train/validation/official-test名单仍为14494/1506/4000。已有4096张补充训练图使训练人口为18590个唯一ID；不新增数据或seed。唯一模型seed为20262020。当前源验证已被反复查看，所有筛选均称源开发结果。历史official test已查看，本次不运行official test或真实域评价，也不据其改结构、窗口或门槛。

标签构建器读取每张原annotation的六类几何：有序针体网格投影点、有序刻度点、相机世界中心、表盘平面H、整数外壳框及刻度角间距。模板按网格顶点数唯一对应三种已冻结scaled模板。它不使用y、pointer angle/rotation、normalized state、factors.r、原模型预测或RGB。

对每个针体顶点j，按原整数标签的cell中心加0.5，通过H得到其z=0相机射线交点w，再在该顶点自身高度z_j处求交：

\[
x_j=w_j+(C_{xy}-w_j)z_j/C_z.
\]

用全部对应顶点的centered二维Procrustes拟合针体object rotation θ。用原有序刻度点与已知间距做复数最小二乘，圆心和半径自由，得到第一条刻度方向φ₀及弧长λ；不假设原刻度高度是新版source的0.002。

\[
\delta=(\phi_0-\theta)\bmod 2\pi,
\qquad
\lambda=(N_{tick}-1)\,\Delta_{tick}.
\]

相机、H、模板与投影点只参与训练标签构建和当前几何诊断。新模型的部署接口仅接收RGB ROI；δ/λ、y及上述几何均不进入模型前向。模型直接估计相对物理状态，不预测需要在推理时输入GT相机的端点。

原14494、补充4096及验证1506的标签均完整valid；18590/1506输出保持原清单顺序，无删行。canonical字段为`source_role=original/supplement/validation`，计数分别14494/4096/1506。标签构建独立于正式训练代码；完整验证详见结果包的[label准备摘要](data/geoprr_v2_physical_interface_20261021/label_preparation_summary.json)。

## 单个空间状态头

```text
RGB ROI768 → 共享MobileNetV4/FPN
                    ↓
        一个空间状态头：π、U(δ)、V(λ)
                    ↓
            完整Q(δ,λ)联合PMF
                    ↓
       冻结物理比值推送 → p(progress)
                    ↓
                 native均值
```

保留普通监督best63的MobileNetV4/FPN共享模块初始化，移除其旧U/M/Q/direct读出头。一个状态头在FPN空间特征上生成两类、每类两个分量的注意力池化结果，结合共享全局图像上下文；它输出两分量混合权重π、每分量δ PMF及λ PMF。新头由同seed初始化，不新增encoder、整图专家、router或独立标定前端。

联合臂使用固定R=2：

\[
Q_{joint}(d,l\mid I)=\sum_{r=1}^{2}\pi_r(I)U_r(d\mid I)V_r(l\mid I).
\]

强对照保留**相同网络输出、全部参数、初始化、标签、前向批和训练预算**，只在组合处打散配对：

\[
Q_{independent}(d,l\mid I)
=\left(\sum_r\pi_rU_r(d)\right)
 \left(\sum_r\pi_rV_r(l)\right).
\]

这个对照用于判断跨变量配对是否必要。若收益可由相同边缘分布的独立组合复制，不主张配对机制有贡献。R=2仅能表示有限相关结构，不能保证每个分量单峰，也不能保证排除所有错误配对。

## 冻结离散网格和完整质量读出

δ为256点圆周网格，坐标`2π*d/256, d=0…255`；λ为256点有序网格`linspace(0,2π,256)`，不把弧长当周期变量。训练δ目标用跨接缝的circular twohot；λ目标用有序linear twohot。原训练有两张几何量化造成raw progress微出界，标签保留原δ/λ，不用y反造或删图；物理读出按下式映射其质量。

对于`0<λ<2π`的网格atom，先在该弧中点附近展开δ，再取比值和固定端点裁剪：

\[
g(d,l)=\operatorname{clip}_{[0,1]}
\left[
\frac{\operatorname{wrap}_{[-\pi,\pi)}(\delta_d-\lambda_l/2)+\lambda_l/2}{\lambda_l}
\right].
\]

把atom全部质量按相邻两个有序progress bin线性分配。弧外质量继续参与端点读出，另记`outside_arc_mass`；不按它删atom、删图或重归一化。

λ=0及λ=2π两列不定义有效的有端点物理弧。其全部质量均匀分摊到256个progress bin，并完整计入`invalid_mass`。不得丢弃这些质量、只统计有效条件分布，或回退到旧模型。最终分布质量与输入Q一致，输出为native均值；没有r8、GT mask、direct回退或按质量筛样本。

比值一般不与取边缘独立化或分别取期望交换。例如两组配对状态可以给出相同读数，打散后却产生额外的交叉比值。这是选择对照的原因，不是算法效果证明。正式训练之前先用冻结标签检查该离散化是否仍满足几何/普通精度门槛；该检查依然不读取RGB或训练模型。

## 两臂训练预算与可比性

| 项目 | joint与independent共同约定 |
|---|---|
| seed | 20262020，不扩seed |
| 共享初始化 | 同一普通监督MobileNetV4 best63的backbone/lateral/refine/appearance_norm |
| 训练人口 | 18590唯一图，原14494＋已有4096 |
| epoch | 65；前5轮共享模块freeze并保持eval，第6轮起共同finetune |
| batch/workers | 32 / 8 |
| 每轮donor/recipient | 22686 / 8192，共30878图像前向 |
| 每轮优化批 | 709；实际成功更新及AMP跳步单独记录 |
| 优化器 | AdamW，shared lr1e−4、head lr5e−4，min lr1e−6、weight decay0.01、gradient clip5、AMP |
| 配对曝光 | 使用同一冻结pair清单、同一photometric变换；donor与有效recipient拼接为一次前向 |
| 目标 | raw soft-pair NLL除以`2*log(256)`，加原归一化y的progress CDF损失；两项权重均1 |
| checkpoint | 同一完整1506源验证native NMAE，严格更小才更新；相同值保留最早epoch |
| 推理 | RGB-only，完整质量native均值 |

65轮计划执行46085优化批和2007070图像前向；成功update、AMP跳步、训练进程wall、epoch wall、参数及峰值由正式日志实测。GT标签构建使用额外结构监督，新头与损失也改变，因此与旧普通监督模型不能称完全等信息或等总训练成本。

旧reader把donor/recipient分成两次前向，新两臂将其concat为一次；BatchNorm上下文存在差异。两新臂之间保持一致，旧reader比较明确保留这个边界。warm start还继承了旧模型已完成的训练历史，不能只计新增65轮来声称从零训练更省。

## 预注册评价和停止规则

先跑joint完整65轮，收齐run/config/history/best/completed与1506逐图完整预测。所有失败保留在分母，error=1。完整结果必须包含NMAE、Acc@1/2/5、>1/>2/>5数量、median/p90/p95、最差ceil(0.1N)平均误差、最大误差、逐图rescues/new errors、全部质量诊断、参数和完整成本。

普通集合C预先固定为native与r8均≤1%FS的1495张；它只用于评价分层，不进入模型、采样或路由。另保护r8普通1497张，防止共同集合遗漏r8已正确的样本。

| 门槛 | 固定条件，单位%FS |
|---|---|
| 共同普通集合mean/median/p90 | ≤0.0972051518 / 0.0564150512 / 0.2246701717 |
| r8普通1497张mean | ≤0.0977505576 |
| 完整1506 NMAE/Tail10 | ≤0.1299504985 / 0.7205782515 |
| 最大误差 | 严格小于native26.6273960471和r8 19.5571362972 |
| 已有严重病例 | 两对照>2并集6张全部≤2；原>5病例全部≤5 |
| 新增错误 | 相对native、r8分别计算；>1/>2/>5三档均为0 |

joint若不满足G1/G2，保留负结果并可跳过independent，不能换r8、扩大seed或修改阈值宣布成功。joint通过后，再用同seed、同预算运行independent；只有完整的两臂实际结果才能支持配对贡献。门槛是开发计算筛选，不是统计显著性检验或准确率保证。

在joint自身通过上述门槛后，机制筛选进一步要求joint的完整native NMAE相对同监督independent至少降低5%，Tail10与>5%FS数量均不增加。该条件在训练前冻结，只决定是否继续研究配对机制，不作为统计显著性或seed扩展授权。

本阶段不运行official test、不扩seed、不恢复旧OOF/router矩阵。后续训练自动模型是否修复重尾，必须由自动输入下的实际逐图结果决定。

## 已有工作的覆盖和待证贡献

[MDPose，UAI2023](https://proceedings.mlr.press/v216/seo23a.html)已经用mixture density建模关键点联合分布；混合形式及“保留配对”的一般思想均非首次。[Leon-Alcazar等，WACV2024](https://arxiv.org/abs/2308.14583)已使用合成结构监督和角度读数网络。因此本候选不能只凭“混合＋角度＋几何”主张创新。

允许保留的候选贡献只有一个：面向物理归一化读数，在同样的两个状态边缘预测能力下，显式保留δ/λ配对并以完整质量推送，能否同时保护普通精度和严重误读。公开的零训练接口为这个问题提供可复现标签与失败检查；RGB完整结果已否决该候选，同参数独立对照因预设gate失败而未运行，当前不能识别配对机制的独立贡献。它没有获得新颖性认证，也没有FFT、Hough替代或射影等变性的理论主张。

## 实现与运行检查

两新臂实测参数均为7,963,234，共享与新头初始张量逐项相同。五项CPU检查覆盖完整概率质量、圆周/有序拓扑、配对相关的必要性构造、端点浮点边界及梯度。真实两张原训练图在两种coupling下均完成GPU前向/反向，FP32概率与有限非零梯度检查通过；共4次图像前向、0次optimizer更新。这是实现检查，不是准确率证据或完整训练批显存测量。正式训练使用已存在项目venv，未改变系统Python、CUDA、conda或其他项目。

## Limitations

GT几何检查比RGB识别提供更多信息，不能作模型性能或上界。密集mesh拟合通过也不证明稀疏端点头通过。模板只覆盖当前已核验源几何，真实仪表几何、遮挡及材质变化仍需独立评价；单seed不提供跨seed样本标准差。反复源验证迭代仅支持开发决策，未来冻结模型后的泛化证据另行建立。当前完整65轮候选已被源开发门槛否决，未建立学习收益；物理标签有效不能替代RGB自动预测证据。
