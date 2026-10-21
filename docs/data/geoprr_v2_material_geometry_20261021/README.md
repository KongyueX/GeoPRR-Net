# 逐高度材料几何：条件接口通过，未知标定链被否决

完整1506张源验证显示：保留每个材料点的真实高度、拟合相机相关干扰参数，可以在给定正确G与刻度间隔的条件下保住普通精度并救回旧严重错误；但把G和间隔改为仅由有序刻度点估计后，名义无扰动读数已明显退化，不能进入训练。当前最佳已完成RGB参照仍是普通监督MobileNetV4的best63，seed20262020。

| 方法／信息合同 | NMAE %FS ↓ | >1 / >2 / >5 | median %FS | p90 %FS | p95 %FS | worst151平均 %FS ↓ | max %FS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 普通监督native，RGB | 0.1479240671 | 11 / 5 / 3 | 0.0571250916 | 0.2393081784 | 0.3892682493 | 0.8974729324 | 26.6273960471 |
| 同模型固定r8，RGB | 0.1299504985 | 9 / 5 / 2 | 0.0573858619 | 0.2371266484 | 0.3810346127 | 0.7205782515 | 19.5571362972 |
| GT G＋已知间隔＋GT材料/刻度对应，不输入GT C | **0.0254598447** | **0 / 0 / 0** | **0.0172678316** | **0.0553919080** | **0.0753629406** | **0.0914318657** | **0.3813899961** |
| 估计G/间隔＋GT材料/刻度对应，不输入GT G/C/间隔/span | 0.6934348068 | 209 / 117 / 41 | 0.0629747681 | 1.4819373874 | 2.8990362153 | 5.2939235016 | 45.6835150719 |

Acc@1/2/5依次为native 99.269588/99.667995/99.800797%，r8 99.402390/99.667995/99.867198%，条件已知标定100/100/100%，未知标定86.122178/92.231076/97.277556%。两条几何链均保存完整1506行，没有删图或使用读数目标选解；目标在全部预测保存后才加入计分。

## 普通样本与新增错误

共同普通1495张由旧native及r8误差均≤1%FS固定。条件已知标定的mean/median/p90为0.025356831/0.017269499/0.055002883%FS，通过全部冻结数值门槛。未知标定对应为0.685264734/0.062972116/1.475604948%FS，新增207张>1、115张>2、40张>5错误。r8独有普通1497张上，两条几何链mean分别0.025340777及0.684605886%FS，参照为0.097750558%FS。

| 候选相对参照 | 全1506逐图better / worse | >1救回 / 新增 | >2救回 / 新增 | >5救回 / 新增 |
|---|---:|---:|---:|---:|
| 已知标定 vs native | 1196 / 310 | 11 / 0 | 5 / 0 | 3 / 0 |
| 已知标定 vs r8 | 1197 / 309 | 9 / 0 | 5 / 0 | 2 / 0 |
| 未知标定 vs native | 656 / 850 | 9 / 207 | 5 / 117 | 3 / 41 |
| 未知标定 vs r8 | 654 / 852 | 7 / 207 | 5 / 117 | 2 / 41 |

旧>2%FS并集6张在两条链中全部回到≤1%FS；这没有抵消未知标定新增的大量错误。

| sample_id | native %FS | r8 %FS | 已知标定 %FS | 未知标定 %FS |
|---|---:|---:|---:|---:|
| sync_10202 | 26.627396 | 16.561826 | 0.070546 | 0.040744 |
| sync_10635 | 2.455890 | 2.455461 | 0.012536 | 0.018992 |
| sync_12540 | 1.364576 | 2.511204 | 0.015617 | 0.361732 |
| sync_338 | 19.577676 | 19.557136 | 0.115550 | 0.325264 |
| sync_3663 | 2.311166 | 3.186789 | 0.000425 | 0.993255 |
| sync_4515 | 17.546070 | 0.904751 | 0.025642 | 0.356633 |

## 物理合同与实际阻断

材料规则在计分前仅凭固定模板冻结：排除精确z分组中≥8点、XY质心范数≤1e−6、原点半径极差≤1e−6、相对协方差各向异性≤1e−6的圆周组，保留全部其他材料顶点。pointer1/2/3分别保留18/12/408点，合计222054个逐图对应；不按半径切掉位于hub下方的材料点。各点使用自己的z，没有同高假设，也没有旧origin/outside接口。

给定metric平面观测w和材料点(X,Y,z)，拟合式为`w(1−bz)=κRθ[X,Y]+t−az`。线性形式仅作初始化；最终使用完整源像素重投影残差，并检查正尺度和全部相机分母为正。相机相关干扰参数a、b由观测拟合，不输入GT相机C。未知标定先用全部有序刻度拟合共同H和间隔，再接材料求解；候选只按刻度/材料重投影残差选择。

理论上正确连续等角刻度在N≥5时可识别间隔；实际1506张N=7–36，全部坐标是整数像素。**理论唯一不等于稳定**：未知链平均刻度拟合RMSE仅0.325695px，仍产生0.693435%FS的NMAE。最大读数错误`sync_8861`为45.683515%FS。另一个数值病态病例`sync_15021`的估计span约0.000415693°、H条件数约3.32×10¹³，保存H与逆G的像素重投影差异最高0.249400px。独立复核确认保存读数及全分母计分，但该链审计状态为**WARN**，不是干净数值通过。

固定初值没有发现near-fit替代解，不构成全局唯一或置信度证明。名义无扰动链已失败，因此没有继续完整1506的点扰动、缺失、假峰压力实验，也没有启动新模型训练或扩种子。下一步需要共同利用真实可观察证据并保留标定不确定性；不能把串行刻度最优标定当作确定真值。

## 观测边界与成本

两条几何链都使用GT材料/有序刻度坐标、正确针型和模板对应，**不是RGB识别成绩或可部署算法性能**。原标注没有可见性、遮挡、材料角点ID或逐图生成revision；完整网格投影不等于可见关键点。

刻度mask同样不是可靠的同面矩形：官方固定源把全部立体cube顶点投影后保留图像坐标极值，再整数化排序。实际73162列表中有70881个4点、68个5点、2213个6点，含2329个重复顶点列表。没有稳定顶面四角、径向边或long/short身份。因此不能将这些polygon直接作为共面矩形或已知径向直线加入标定。

两个完整CPU解码进程分别16.599261s和25.165829s，均0 CNN前向、0模型优化更新、0新源图、0新seed；时间不含标注导出、独立审计、归档和未来检测。几何求解没有学习参数。SciPy1.15.3仅安装于项目venv并使用`--no-deps`，NumPy保持1.26.4。普通RGB参照参数7695954，不把几何条件计时当作camera端到端速度。

本阶段仅为源验证开发证据，未验证真实域迁移，也未确认论文级创新。G、点身份和标定监督边界不能混入自动模型排行榜。

## 文件与复算

- [完整1506宽表CSV](predictions.csv.gz)、[同内容JSONL](predictions.jsonl.gz)：只含规范化读数、误差、固定集合、材料数量及派生标量；不含原图像点、模板XYZ或G/H/C矩阵。
- [baseline-only reference](reference.csv.gz)：原RGB参照读数和误差，用于独立计分。
- [完整指标](metrics.csv)、[摘要和冻结门槛](summary.json)、[逐阈值ID变化](threshold_changes.csv)、[配对变化](paired_changes.json)、[旧严重6张](severe_cases.csv)、[各方法worst151](worst151.csv)。
- [完整协议](protocol.json)、[冻结材料索引](material_selection.json)、[目标盲观测库存](observation_inventory.json)、[刻度形状观测库存](tick_shape_observation_inventory.json)、[成本](costs.json)。
- [已知标定独立复核](known_geometry_independent_verification.json)、[未知标定独立复核WARN](observed_calibration_independent_verification.json)、[锁定预测后的病态诊断](conditioning_posthoc.json)。
- [物理与部署门槛说明](../../GEOPRR_V2_LOCAL_PHYSICAL_PREFLIGHT_CN.md)。

以下命令需要研究者持有原始私有几何及固定模板，公开包不提供它们。设置`PRIVATE_GEOMETRY`为原始目标盲1506几何JSONL，`PRIVATE_TEMPLATES`为三模板快照，`PRIVATE_SPACING`仅用于第一条条件链；`OUTPUT_DIR`须为不存在的新目录。当前结果无需重复运行。

```sh
python -m experiments.audit_geoprr_v2_material_condition \
  --geometry "$PRIVATE_GEOMETRY" --templates "$PRIVATE_TEMPLATES" \
  --selection docs/data/geoprr_v2_material_geometry_20261021/material_selection.json \
  --spacing "$PRIVATE_SPACING" --output "$OUTPUT_DIR"
python -m experiments.summarize_geoprr_v2_material_geometry \
  --predictions "$OUTPUT_DIR/predictions.jsonl" \
  --reference docs/data/geoprr_v2_material_geometry_20261021/reference.csv.gz \
  --output "$SCORE_OUTPUT_DIR"
```

第二条链从有序刻度估计G/间隔，命令没有spacing参数：

```sh
python -m experiments.audit_geoprr_v2_observed_calibration \
  --geometry "$PRIVATE_GEOMETRY" --templates "$PRIVATE_TEMPLATES" \
  --selection docs/data/geoprr_v2_material_geometry_20261021/material_selection.json \
  --output "$OUTPUT_DIR"
```

代码：[逐高度材料核](../../../experiments/geoprr_v2_height_profile.py)、[有序刻度标定](../../../experiments/geoprr_v2_tick_calibration.py)、[条件解码入口](../../../experiments/audit_geoprr_v2_material_condition.py)、[未知标定入口](../../../experiments/audit_geoprr_v2_observed_calibration.py)、[保存结果计分器](../../../experiments/summarize_geoprr_v2_material_geometry.py)。
