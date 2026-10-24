# v22真实相机roll源覆盖与单图直接读数

**1152组／4608张新增真实源图已完整生成并通过准备，新RGB零更新运行见证通过，正式源训练正在运行；尚无完整外域性能结论。** 模型回到完整fresh缓存ImageNet MobileNetV4 semantic1280→point→sigmoid，单张256 ROI一次CNN，没有orbit均值、几何router或新loss。预算不要求与旧实验相同，源学习决定60–160完整epoch。

旧18590 unique源图（v11端点保留）不改，新增1024原训练布局×4状态＝4096真实相机roll RGB。前128组使用当前[0,.005,.02,1]端点，余896沿用原quartet；每组roll在[-180,180)预先独立采样并共享四状态，以相机局部光轴quaternion组合实际渲染和重新投影，不旋转旧RGB冒充新物理图。另128新layout seed生成512源roll验证，同一现有HDR背景，不称背景／域holdout；原1506旧scene holdout保持，新val采用独立layout／ID／group／generated-seed合同，绝不伪造scene通过旧split guard。

新4608张图的物理准备复用原768／192完整geometry_targets，每组16次Q×M×U cross-state检查，保留min mass .15／max error .01；只检查新生成数据，不重审旧源，也不声称64网格Q通过。训练仍256 RGB／scalar，不输入GT几何。

每epoch旧18590一次＋新4096各4次＝34974次呈现／547个batch64，unique22686，新roll曝光46.8462%；重复是同一固定RGB的独立光度抽样，不是四个新物理roll。归一化exact-start曝光640／34974约1.83%，旧v16为128／18590约0.6885%，不必是绝对物理零读数。当前仅有1个可用HDR和三种原针型，没有新增背景或连续针形覆盖。

源score固定为0.5×(原1506 mean NMAE＋独立512 rollval mean NMAE)，最早严格最小选best。fresh AdamW初始LR3e-4、原SmoothL1 beta.05；ReduceLROnPlateau factor.3／patience8／threshold0 abs／minlr1e-6。最少60轮后，距严格新best≥20轮停止，counter从epoch1开始，因此最早60轮结束；max160兜底。scheduler降LR不重置earlystop。实际source结束后才各一次冻结best评Industrial1395／52、RF100151／35，last只源记录。目标严格不参与训练、统计／BN适配、checkpoint、LR、停止或超参数选择，失败保留完整分母error1。

正式每epoch34974训练图／547次反传尝试、2018验证图／32次调用；60轮下界总2221066图／34765次CNN，160轮上界5920266图／92665次CNN，均含最终1546目标图／25次调用。实际epochs／updates／AMP skips、生成4608新RGB与物理核对、训练／推理时间和峰值内存分别记录，不能报预算已完成。完整结果后再通知；不要求额外公平对照阻断性能实验。

- [完整协议](../../GEOPRR_V2_ROLL_SOURCE_ZERO_SHOT_PROTOCOL_CN.md)
- [与训练器实际CONFIG对齐的配置](config.json)
- [显式公开allowlist](public_allowlist.json)
- [完整新源准备与生成成本](source_completed.json)
- [实际新RGB零更新运行见证](runtime_preflight.json)

## Limitations

新数据、重复采样、端点曝光、新源验证和adaptive stopping共同改变，不作单camera因果声明。新rollval只在新布局／generated seed／ID／group层面独立，共享HDR，不能冒称原14494历史seed全链已认证或新域盲测。源score改善不保证外域收益；已查看两外域和单seed不称显著性或普适机制。原14494／1506／4000、旧模型与失败全部保留，RGB／权重／原几何／完整计划及合同不公开，不删旧资产。

16项针对性CPU检查及实际生成计划入口检查通过。完整新增源数据保存的检查聚合为min GT support 0.2452455163、max composed error 0.8006334991%FS、max光轴误差2.980232239e-7；源生成supervisor阶段19265.214674秒，与模型运行见证及正式训练成本分开。新运行见证实际3次CNN／8图／1次反传／0更新，完整8435793参数／464 state严格加载、77 BN counters源训练+1／eval额外0、两套验证256输入identity。只有训练及两套冻结外评、唯一完成阶段CPU核对结束后才给出性能结论。
