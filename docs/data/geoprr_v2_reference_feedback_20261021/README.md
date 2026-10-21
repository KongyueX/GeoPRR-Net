# 参考权重能量反馈：实现准入与正式启动

feedback正式训练于北京时间2026年10月21日17:39:34启动，完整65轮结果待完成。只改变P／B原始匹配能量对参考A的全局归一化反馈，无新增参数、loss或标签；同参数local删除对照在feedback完整通过原G1后才运行。协议见[正式合同](../../GEOPRR_V2_REFERENCE_FEEDBACK_PROTOCOL_CN.md)。

两臂均从完整v5 best57的496个状态tensor初始化，逐tensor相同，模型7,461,060参数。唯一seed20262020。每轮18590 unique、22686 donor＋8192 recipient、30878图像前向、709执行批；65轮，5轮冻结共享模块，第6轮起共同微调。完整1506 NMAE选best，最早平局。

## 新增实现检查

六项CPU小矩阵检查通过：全局三角色概率与分解一致、two-hot NLL抵消、同参数及相同条件P／B、循环roll与全质量、row offset反例、全部角色参数梯度。实际7次未训练组件特征前向、14个特征样例、1次反传；0骨干图像／0optimizer／0GPU调用。该进程0.214882s，只代表算术测试成本。

实际RGB准入固定sync_3172与sync_10，各在两臂前向一次，共4张CNN图像前向、1次feedback头反传、0优化。全1506监督合同及ID／group／float32 target与best57一致；全权重及buffer相同、共享冻结统计不变、loss／梯度有限、概率与CSV接口通过。最大概率质量偏差1.192093×10⁻⁷，local两图与best57保存读数最大差7.152557×10⁻⁷。

该runtime进程7.474959s，记录的两图峰值allocated约164／182MB，只是小批冻结共享的准入成本，不能代替正式训练峰值或推理速度。初始feedback与local输出、loss可以不同，不能据此当作学习效果。

best57比较器用完整1506自身比较通过，1495／1497普通人口一致、救回和新增均0，0新网络前向。既有完整1506教师标签合同沿用，未重跑GT几何或best57完整PMF诊断。

## 阶段规则

feedback65完成后按原11项G1逐图计分；未过保留负结果并跳过local，通过才local65。G2还需反馈相对local NMAE至少降5%、tail／>5不增加，以及相对best57总体／普通精度和新增严重错误保护，详见冻结协议。旧v5 product未运行，其他seed与official test均禁止。

启动核查父123362、子123365命令匹配，唯一GPU模型为该训练；epoch1预热首批32 donor＋11 recipient、loss有限。启动时0轮完成，当前实时进度以服务器history／latest和实际进程为准。

## Limitations

全局归一化已有先例，行partition或query范数可能强化错误参考；新候选效果和机制收益尚无完整结果。单seed、反复查看源开发筛选不代表显著性、盲测或真实迁移。准入只证明实现与训练接口可用，不证明准确率成功。

## 文件

- [算子检查](operator_checks.json)、[实际RGB准入](runtime_preflight.json)、[best57比较器](comparator_check.json)、[启动状态](launch_state.json)。
- [实现及停止协议](../../GEOPRR_V2_REFERENCE_FEEDBACK_PROTOCOL_CN.md)、[结构与匹配对照](../../GEOPRR_V2_REFERENCE_FEEDBACK_DESIGN_CN.md)。
