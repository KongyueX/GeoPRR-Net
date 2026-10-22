# GeoPRR V2：冻结模型的外部数据评估

日期：2026-10-22。用户授权使用模型测试其他几个数据集。本阶段排在唯一的v8 matched global65完成之后，不并行占用当前GPU，不增加训练、loss、模块、模型seed或训练轮数。

## 模型与选择

首轮锁定已完成并保留的v7 full best64：`runs/geoprr_v2_tail/iteration_v7_role_triangle/readers/full_s20262020/best.pt`，模型seed20262020、epoch64、7,461,316参数，源开发1506 NMAE为0.0591462310626%FS。它是配对守恒扩展前的完整结构模型；本次先检查这一既有模型能否迁移，不再展开多模型矩阵。各臂best及v8机制比较仍按原合同报告，不用目标数据选权重、阈值、温度、归一化、ROI或读数方式。实际加载时核对原配置及checkpoint元数据，执行一次clean原生读数评估。

若配对守恒相对匹配global没有增量价值，不将其作为论文主贡献；外部测试不用于挽救或改写匹配机制结论。当前阶段只输出冻结模型的外部描述性结果。

## 人口与输入

| 数据集 | 固定人口 | 计分与输入边界 |
|---|---:|---|
| Industrial-1395 | 1395张、52组；原Test-A 434／Test-B 814／External-ROI 147 | 原有仪表ROI RGB，标签已有normalized_progress；总人口及三个子集分别报告 |
| RF100-VL | 151张、35组 | 原有ROI RGB；目标由四点标注离线派生，明确标注派生目标来源，不称官方标量读数benchmark |
| RPM-10K单针子集 | 原冻结规则选出的1797张，来自官方test 2000行 | reading/range归一化；先确认已有detector ROI与完整sidecar来源，失败保留全分母；ROI合同未确认前不推理 |

服务器已确认Industrial与RF100各自input_manifest／labels的ID集合相同、完整ROI文件存在，未运行新模型。2026-10-22的只读盘点确认RPM原test有2000行，按既有规则得到1797个唯一合法单scalar样本，所选原图全部存在；当前未找到该人口的detector ROI输入清单或完整失败sidecar。test_drive_index只有下载id／size，不是检测结果。旧1775成功／22失败属于旧Windows记录，不能当作当前服务器产物。先推进Industrial与RF100；RPM须恢复原ROI产物和来源，或在推理前另冻结一次ROI准备合同及1797全分母失败表。当前不启动detector，不把原图直接改当ROI。

Pointer-10K现有固定单针438图只有针的方向点和bbox，缺少本模型可直接计分的标量读数／量程；虚拟P相位也不是物理针尖。本阶段不构造其NMAE或把虚拟角色冒充方向输出。原SyncG officialtest不复跑。

## 执行与报告

推理仅原ROI RGB，使用所选模型原768／192处理、ImageNet归一化、eval模式和原生全质量读数。不得读取GT H、center、SPD、起止方向或针尖作为模型输入。采用对应结构Reader严格加载checkpoint，不能将新checkpoint协议改写后塞进旧模型loader。

0反传、0optimizer、0目标训练／适配。首轮仅clean单图，不扩退化、路由、窗口或模型矩阵。完整人口逐图保存prediction／target／valid／failure；图像或前端失败按原error=1计分，不删失败样本。报告NMAE%FS、Acc@1／2／5、阈值数量、分位、最差10%、max及ID、组和子集结果、实际CNN图像前向／wall／峰值。

Industrial／RF100已有历史结果被查看；本次不称全新盲测。ROI来源、RF100标注派生目标及RPM detector来源分别披露。只有人口、输入与计分均一致的历史结果才可作比较，不将不同辅助条件混成一个排名。外部结果不反向用于当前方法选型或追加训练。

RGB、权重、原始几何／像素／矩阵与private trace仅保留server／private；公开仍用显式标量allowlist。
