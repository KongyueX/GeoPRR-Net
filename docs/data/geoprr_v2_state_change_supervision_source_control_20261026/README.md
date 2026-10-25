# Source state-change supervision control

已冻结前瞻协议，新3代码与12项必要synthetic CPU检查已完成；尚未生成全source辅助目标、构造真实模型或运行GPU。不是正式V32或新的泛化结果。

问题：正确state对应的image-derived空间残差监督，能否在保留原RGB scalar推理路径时改善未见source布局，并优于相同训练过程中的固定循环错配对应控制。新增的是source图像派生监督范围，不能继续称原scalar-only合同；目标不是针体/刻度/visible mask或GT几何。

先在CPU检查全部1374个既有训练quartet并构建5496个16x16目标，任一对齐/变化/退化/对应差异门槛失败即整体停止，不删quartet、不启动模型。只有准备通过，才一次source0opt见证，再fresh正确对应15E；该arm通过完整native+CF联合门槛后，才fresh循环错配15E，后者native和CF各固定计分一次。最后统一判定正确arm相对identity15与错配arm的mean/tail/max/>5及各自准确集保护，不能挑checkpoint或改超参救过。

完整定义见[protocol.json](protocol.json)和[中文协议](../../GEOPRR_V2_STATE_CHANGE_SUPERVISION_SOURCE_CONTROL_CN.md)。配置与NEWCPU证据见config.json/cpu_checks.json；实际执行尚未开始，这里没有模型成绩。source03与真实Industrial/RF均不在训练、辅助标签构建或控制模型计分范围。

## Limitations

一个源quartet的局部变化观测仅为问题依据，不证明全人口支持、可读性或物理mask语义。残差可能包含阴影、遮挡、JPEG及其他state残影。使用已查看source开发集和人工CF，单seed不作新盲测、显著性、因果机制或真实泛化结论；历史pixel/fullweight与raw-label派生资格不在此重认证。
