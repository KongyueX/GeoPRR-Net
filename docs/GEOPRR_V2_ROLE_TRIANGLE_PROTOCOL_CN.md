# 三角色内容关联：正式实施协议

本协议对应[三角色关联设计](GEOPRR_V2_ROLE_TRIANGLE_DESIGN_CN.md)。保留v6反馈best48全部权重及正式结果，只新增指针P—终点B双线性因子；对照用该因子自身P／B边缘乘积。两臂共用精确三角色归约和完整质量读数。原11项G1保持，best48保护单独报告。

full于北京时间2026-10-22 02:54:24正常完成65轮，最终best64；原G1通过10／11、best48保护5／6，因新增错误保护失败而拒绝，separable按冻结规则跳过。训练代码9909333；父157356／子157359正常退出。原资源准入与全部最终记录见[公开包](data/geoprr_v2_role_triangle_20261021/README.md)。

## 固定合同

| 项目 | 固定值 |
|---|---|
| 输出／控制器 | runs/geoprr_v2_tail/iteration_v7_role_triangle；experiments.run_geoprr_v2_role_triangle |
| 训练入口／协议 | experiments.train_geoprr_v2_role_triangle；geoprr_v2_role_triangle_source_only_v1 |
| 候选／删除 | association=full／separable；两臂均完整三角色联合读出 |
| 模型seed／参数 | 唯一20262020；两臂均7,461,316参数 |
| 初始化 | v6 feedback best48，496项旧state逐tensor加载，仅新增state_head.Wpb置零 |
| 周期 | 每臂65轮，前5轮冻结共享模块及统计，第6轮共同微调 |
| 人口 | 原14494／1506／4000冻结；沿用4096补充源图，无新渲染 |
| 每轮曝光 | 18590 unique；22686 donor＋8192 recipient＝30878训练图像前向；709执行批 |
| 尺寸／batch | 输入768、field192；donor batch32，recipient可追加32，最大有效batch64；val16；8 workers |
| 优化 | 原AdamW／cosine；共享lr1e−4、头lr5e−4、min1e−6、weight decay .01、clip5 |
| 监督 | 三角色联合CE／(3log256)＋原field L1＋原progress CDF，第1轮起权重均1 |
| 数值／best | AMP骨干、FP32头／loss、FP64标签；完整1506 native NMAE严格最小、平局最早 |

仅name与model.association在两配置间不同。state_head.Wpb纳入原头参数组，第1轮即可更新；原共享模块冻结规则不变。有效optimizer更新和AMP跳过按两臂真实执行分别记录，不将相同计划写成完全相同有效更新。旧训练入口默认依赖不变，新wrapper只注入新reader、正确联合loss和full_best48初始化。

## 网络、标签及读数

完整因子F由256×256的P—B双线性能量全归一化产生；separable仅用同一F的两个边缘乘积Fsep。新增矩阵16×16，无新增标签、loss、温度、窗口、entropy门控、fallback或模型seed。零W时两臂都退化到保留的best48全局star模型。

三角色联合CE包含原star交叉熵及logZcorr−E_label(logF)修正；separable替换为logFsep。不能调用旧star-only角色NLL冒充新联合CE。全256³状态精确分块求和，分块只改变临时存储；不建立完整稠密T缓存、不裁剪状态或质量。参考entropy／max必须取真实三角后验A边缘，不能把star A标为新后验。CSV的counterfactual_reading为同checkpoint新相对J的两个边缘乘积读数，仅作诊断，不能替代本次separable训练。

推理只输入RGB ROI。H／span／y与虚拟角色phase沿用既有训练／计分语义；经验径向query不是物理针体，虚拟P不是物理针尖，不增加同高、角色互斥或隐藏身份条件。span零／full-turn alias质量仍uniform进入progress，其余沿用原中点展开、clip及two-hot运输。

## 准入与冻结

准入记录保存至diagnostics/role_triangle_preflight_v1_20261021/runtime_preflight.json，包含两份实际配置、逐tensor同初始化、0optimizer、原教师门槛及真实RGB／资源检查。控制器要求status=passed、same_initial_parameters=true、optimizer_steps=0、teacher_numerical_gate_passed=true，且保存配置与正式配置完全相同。

实现检查保留两个边界：少量真实RGB的W=0读数／loss须重放best48，冻结容差1e−5；实际最大donor32＋recipient32有效batch64须完成共享finetune反传，检查全部必需梯度有限、概率质量、真实峰值显存和stepwall。warmup中共享统计保持冻结，W可训练。少量图可以重复用于资源压力，但unique图数和实际图像前向必须分开记账，不把资源检查当新模型精度。

若资源不满足，先修精确实现并保留失败成本；不静默减batch、曝光、状态或原门槛。资源检查使用何种tile、有效batch和反传方式须落盘，再让两配置同步冻结。控制器在配置／准入不一致时拒绝启动。

## 阶段与门槛

full先完成65轮，按完整1506 native NMAE选best。原11项G1使用既有[反馈正式协议](GEOPRR_V2_REFERENCE_FEEDBACK_PROTOCOL_CN.md)与compare_to_ordinary的原10项RGB检查，G0沿用既有标签rawmax≤1%FS结果；任何数值、普通人口或阈值均不改写。

另单列best48保护：总体NMAE、共同普通1495 mean／median／P90与r8普通1497 mean均不差best48；相对best48无新增>2／5错误。该保护与原11项分别输出，不能将两者混成新的原G1。

full只有同时通过原G1及best48保护，控制器才运行本次matched separable65。任一未过即归档完整负结果、逐图与成本并跳过separable。旧v6 local、v5 product及旧实验矩阵保持未跑，不补对照、其他seed或officialtest。

两臂完整后G2机制筛选要求full相对separable NMAE至少降低5%，最差151张均值与>5数量不增加，同时保留best48保护。未跑的训练对照写为未跑；同checkpoint删除、GT条件干预和量化教师结果均不能替代正式匹配对照。

## 归档与边界

每臂收齐protocol／config／run／history／best／completed及完整1506 ID／group／target／prediction／valid。报告原native／固定r8／best48逐图比较、Acc1／2／5、>1／2／5、median／P90／P95、worst151／max、旧严重6、全部救回／新增与普通1495／1497。执行批、有效更新、AMPskip、训练／验证前向、完整wall和真实峰值显存分别记账。

run.json／data_contract包含原始几何，仅private归档。公开使用显式allowlist，只上传本任务代码与规范化标量／逐图读数，不上传RGB、权重、vendor、.aris、原始H／C／XYZ／像素或矩阵缓存。仅现有main与项目venv，不改系统或其他项目、不购买算力或向第三方发消息。

启动或授权恢复前source experiments/geoprr_v2_env.sh并核对真实进程与GPU；父子命令、pipeline.pid、queue_state、原step日志及history／latest共同判断活性。best不更新不代表停滞。活子不重复启动，已完整子任务可按原控制器接纳；故障先读原日志／真实命令，同output修复真实问题，保留partial及不利结果。实质方法／配置／名单变化另存新实验。

## Limitations

本次为唯一seed、反复查看源开发的结构筛选。标准三角图模型与双线性匹配不构成新定理；只有匹配对照及普通样本保护通过后，才支持本任务的关联机制收益。历史officialtest已查看，不据其调参或称盲测／真实迁移，不报跨seed标准差。GT条件瓶颈支持检验该结构，不保证实际RGB学习能修复2810／3663。
