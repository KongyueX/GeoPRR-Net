# 三角色内容关联：资源准入与正式启动

完整模型、匹配队列与资源准入已完成，full候选于北京时间2026-10-21 **22:17:11** 启动65轮正式训练，代码9909333。父157356、训练子157359命令与唯一GPU任务一致；其8个DataLoader worker不是重复模型。22:18:39核查仍在epoch1头部warmup，尚无完整轮，首批32donor＋11recipient、loss有限。启动读数不是方法效果，稳态ETA待第6轮后多轮耗时再估。

每臂7,461,316参数：共享7,431,248、头30,068。实际496项best48旧state逐tensor精确迁移，仅新增state_head.Wpb的256个元素置零；两臂初始化完全一致。候选用完整P—B因子，对照保留同一因子的两个边缘并删除关联。原三项目标、seed20262020、输入768／192、batch32＋recipient最多32、65轮／5warmup及每轮30,878前向／709执行批保持。

## 实际RGB准入

固定2810与3663两张unique RGB；每臂先2图头部warmup反传，再重复两图至最大有效batch64做共享finetune反传。两臂共 **132个CNN图像前向、4次反传、0optimizer更新**，没有重复完整1506模型前向或几何教师检查。

| 检查 | full | separable |
|---|---:|---:|
| W=0与best48保存读数最大差 | 2.123415e−6 | 2.115965e−6 |
| W=0与原star目标最大差 | 0 | 0 |
| 最大batch64概率质量偏差 | 2.980232e−7 | 2.980232e−7 |
| batch64峰值allocated bytes | 22,255,050,240 | 22,271,958,528 |
| batch64峰值reserved bytes | 28,194,111,488 | 28,194,111,488 |
| batch64前向＋反传记录秒数 | 11.011865 | 0.487467 |

全部必需梯度有限，warmup共享统计／参数保持冻结，新W在头LR参数组可训练。tile16通过并在两配置同步冻结。整个准入进程23.224318s。

两臂按顺序执行，首批／缓存状态不同，上表耗时不能用于两臂推理或训练速度比较；也不能外推65轮耗时。零优化准入没有分配Adam矩状态，所列显存不是最终训练峰值。实际服务器为NVIDIA RTX PRO6000 Blackwell Server Edition，总显存查询97887MiB；资源评估使用该实际设备，不以早期假定的32GB代替。

## 新增CPU与队列检查

纯算子既有3项检查保留，不重复；新2项CPU集成检查通过，覆盖真实三角A后验／联合CE／梯度和模型参数／初始化接口／头LR／冻结边界。实际为2次特征头图像调用、1次三角归约、2次梯度查询，0CNN／0GPU／0optimizer，进程2.661064s。CPU模拟权重加载与上述实际best48权重迁移分开记录。

匹配配置只在name和association不同。既有1506缓存的best48自比较通过6项追加保护；以v5best57回比较则正确拒绝并检出3172新增>5。该比较器检查没有CNN／optimizer。只读代码复核未发现当前合同下的实质问题，不代表训练结果或新颖性审定。

## 正式阶段与边界

当前只运行full。它完成65轮后，按原11项G1及单列best48保护计分；只有两者均通过才启动本次separable65。未过则保留完整负结果并跳过，不补旧v6local／v5product、其他seed、officialtest或loss矩阵。机制需相对separable NMAE至少下降5%、tail／>5不增，同时保持best48保护。

保留的v6best48正式NMAE0.0636129929%FS、max4.558325186、>1／2／5=5／2／0及其9／11门槛状态不变。新候选尚无完成验证成绩，不能声称2810／3663已获救，也不能把普通三角图模型当论文创新认证。全部为单seed、反复查看源开发的筛选，历史test、真实迁移与显著性结论保持未验证。

## 文件

- [实际RGB与资源准入](runtime_preflight.json)、[新增CPU集成检查](reader_cpu_checks.json)、[既有纯算子检查](operator_checks.json)。
- [启动状态](launch_state.json)、[正式协议](../../GEOPRR_V2_ROLE_TRIANGLE_PROTOCOL_CN.md)、[结构与匹配对照](../../GEOPRR_V2_ROLE_TRIANGLE_DESIGN_CN.md)。
- [导致本候选的完整1506冻结诊断](../geoprr_v2_feedback_failure_diagnosis_20261021/README.md)。

公开仅允许字段中的配置、ID、标量与成本；完整run／data_contract及原始几何仅private保存。训练结束后才补充完整逐图、history、best／completed、门槛、救回／新增、普通人口与完整成本。
