# 参考权重能量反馈：正式实施协议

本实验仅检验P／B原始匹配能量的行log-partition反馈参考A，详见[设计](GEOPRR_V2_REFERENCE_FEEDBACK_DESIGN_CN.md)。以完整v5 best57初始化；local删除对照采用相同joint读出，只恢复独立A归一化。无新增参数、loss、标签、温度或entropy路由。

## 固定合同

| 项目 | 固定值 |
|---|---|
| 输出／控制器 | runs/geoprr_v2_tail/iteration_v6_reference_feedback；experiments.run_geoprr_v2_reference_feedback |
| 训练入口／协议 | experiments.train_geoprr_v2_reference_relation；geoprr_v2_reference_feedback_source_only_v1 |
| 模型seed／参数 | 唯一20262020；两臂均7,461,060参数 |
| 候选／删除 | reference_normalization=global／local；两臂均joint读出 |
| 初始化 | v5 joint best57，全权重及buffer严格加载，两臂逐tensor相同 |
| 周期 | 每臂65轮；前5轮冻结共享模块与统计，第6轮共同微调 |
| 人口 | 原14494／1506／4000冻结；沿用既有4096补充源图，无新渲染 |
| 每轮曝光 | 18590 unique；22686 donor＋8192 recipient＝30878图像前向；709执行批 |
| 尺寸／batch | 768／192；train32、val16；8 workers |
| 优化 | 原AdamW、cosine；lr1e−4／5e−4，min1e−6，weight decay .01，clip5 |
| 监督 | 原角色NLL／(3log256)＋field L1＋progress CDF，自第1轮权重均为1 |
| 数值／best | AMP骨干、FP32头／loss、FP64标签；完整1506 native NMAE严格最小，平局最早 |

配置仅name和model.reference_normalization不同。真实执行批、有效更新及AMP跳过分别记录，相同计划不冒充相同有效更新。

## 实现与准入

feedback在行softmax前计算eP／eB的logsumexp，A=softmax(uA+zP+zB)；local仍为softmax(uA)。P／B、循环相对索引、joint及全质量读数相同；span0／fullturn alias仍均匀进入progress，无r8、direct或mass删行。

六项新增小矩阵检查覆盖全局概率、two-hot NLL抵消、匹配参数／P／B、循环roll／质量、row offset反例和梯度。既有完整1506标签／物理合同沿用，不重跑教师或best57全量诊断。

RGB准入固定sync_3172、sync_10，两臂各前向一次共4张CNN图像前向；feedback一次头反传，0optimizer。核对完整best57初始化、共享冻结统计不变、全1506监督合同／ID／group／target、有限loss／梯度、概率质量与CSV接口。local两图读数与best57相差须≤1e−5。两图准入不用于判断候选准确率。

## 阶段与门槛

feedback先完整65轮。原11项要求不变：G0沿用既有标签rawmax≤1%FS；完整1506有效；NMAE≤0.1299504985%FS、worst151≤0.7205782515%FS、max严格<19.5571363%FS；旧严重6均≤2%；相对普通监督native／r8无新增>1／2／5；共同普通1495 mean／median／P90≤0.0972051518／0.0564150512／0.2246701717%FS，r8普通1497 mean≤0.0977505576%FS。

feedback未过G1即归档负结果并跳过local；通过后才同seed local65。这个local是新反馈删除，旧v5 product保持未运行。其他seed与official test均禁止。

G2机制筛选另外要求：feedback相对local NMAE至少降低5%，worst151及>5数量不增加；feedback NMAE不差于best57、相对best57无新增>2／5；共同普通1495 mean／median／P90及r8普通1497 mean不差于best57。这些是对当前更强基座的额外保护，不替代或放松原11项。每臂均保存与native／r8／best57的全部逐图和阈值变化。

## 执行边界

启动前source experiments/geoprr_v2_env.sh，核实真实进程与GPU。父子PID、queue_state、当前step原日志及history／latest用于活性核验；best不更新不代表停滞。保留独立输出、代码快照、protocol／config／run／history／best／completed与完整1506逐图；run含原始几何，只private，公开采用allowlist。

不改live模型、配置或名单；真实故障先核实原日志／活进程，必要针对性修复并同output恢复。保留partial与不利结果，实质方法变更新实验。仅已有main与项目venv，不改系统／其他项目、不发第三方消息。

## Limitations

全局归一化有先例，row partition和query范数可能强化错误参考，不等于正确性或置信度；P／B给定参考后的条件独立仍在。GT条件效果不保证新RGB头成功。G2仅单seed、反复查看源开发筛选，不代表显著性、盲测或真实迁移；两臂同时改善也不能自动归因反馈。
