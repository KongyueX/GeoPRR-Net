# v21完整语义直接条件角色能量

**独立源域结构与60轮预算已冻结，6项新CPU检查通过，实际构造核实8823492参数／503 state／263个参数tensor；源RGB运行见证与启动尚待完成，没有新模型性能。** 本轮在完整fresh v20的502项base state之后增加一个全零、无bias的Linear1280→3×16；每图semantic query与原角度角色token点积后加入unary，原local共享A条件联合与全质量读出不变。实际新增61440参数／1 state。

query全零时原502项初始化与初始reading必须匹配同seed fresh v20，不加载训练好的旧checkpoint。新query在固定参数下按每图RGB计算，不是目标适配；没有新scalar point、fusion、router、CDF或GT输入。P仍为半径1虚拟盘面状态点，源H／span／U／M和角色只进loss。固定loss沿用v20：SmoothL1(reading,y,beta=.05)+0.1×(roleNLL/(3log256)+fieldL1)。

只用18590源训练／原1506验证、原256 LINEAR／ImageNet／光度与seed20262020，fresh AdamW、同两段30cos LR、60完整epoch；最早源NMAE最小best在全部60轮结束后才各一次冻结Industrial1395／52和RF100151／35。目标不参与训练、总体统计拟合、BN适配、checkpoint或训练超参数选择，完整失败分母error1。计划正式1207306 CNN图／18925调用，不报预算已完成；额外query／关系读出计算、时间和峰值内存另实记。

[v20完成报告](../geoprr_v2_semantic_relation_20261023/completed/report_cn.md)显示源改善而两外域退化，只提供本轮结构问题的动机，不证明新query有效。既有源解析和[理想角色映射兼容性](../geoprr_v2_semantic_relation_20261023/source_role_mapping_feasibility.json)复用，不重做旧GT或新建重复数学检查。outside_arc_mass不能硬门控正确性。

- [冻结协议](../../GEOPRR_V2_SEMANTIC_ROLE_ZERO_SHOT_PROTOCOL_CN.md)
- [新CPU检查与实际模型计数](cpu_checks.json)
- [公开allowlist](public_allowlist.json)

## Limitations

零query时新W可有非零梯度，但不能声称初始encoder已从新query路径获得非零梯度；学习效果须实测。相同base初始化不等于全部随机步骤相同，常规语义角色匹配不构成新理论。两外域已查看、单seed不称新盲测或显著性；全部旧模型、原G1／anchor／65轮与负结果保留，原14494／1506／4000和普通／严重名单不变。无收益不自动升最终方法、不延长或追加seed。
