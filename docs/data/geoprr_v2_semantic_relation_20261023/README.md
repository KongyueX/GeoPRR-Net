# v20完整语义条件关系读出

**独立协议与60轮源域预算已冻结，模型实现及4项新CPU检查通过，实际构造核实8762052参数／502 state；实际源RGB零更新见证也已通过，唯一源训练于北京时间2026-10-23 17:41:09启动，暂无完整性能结果。** 本轮保留完整ImageNet MobileNetV4语义与多尺度FPN，用semantic1280条件预测校准和参考角色联合，再将联合全部质量送入原CircularProgressPushforward读数。没有直接标量point、融合router、四方向轨道或teacher。

模型forward只接收RGB；H／span、解析U／M和半径1虚拟A／P／B标签只供源loss，P不是实体针尖。固定loss为SmoothL1(relation_reading,y,beta=.05)+0.1×(roleNLL/(3log256)+fieldL1)，不加CDF。实际完整模型有262个参数tensor，其中非CNN为31个；CPU检查不代表实际源训练或泛化结果。

只使用18590源训练／原1506源验证，fresh缓存ImageNet初始化、256 LINEAR／ImageNet、原光度、seed20262020，固定60完整epoch；只按源部署reading NMAE选择最早best。全部60结束后才各一次冻结Industrial1395／52与RF100151／35外评，两个外域不得用于训练、统计拟合、BN适配或checkpoint／超参数选择。计划正式总1207306 CNN图／18925调用，关系读出增加计算须另记录；不能把预算当作已完成。

现成的一次源CPU映射检查覆盖18590行／291批，11.688346秒，0 RGB／CNN／反传／更新。共同A条件联合映射均值误差0.002771582%FS、max0.430768728%FS，0行超过原1%FS界限；没有模型权重、目标数据或旧Gaussian Q。这个结果只证明理想源标签与原读出的兼容性，不证明模型可学习或泛化。端点可同时具有约0.999997的outside_arc_mass和准确读数1，不能由该质量硬门控准确率。

- [冻结协议](../../GEOPRR_V2_SEMANTIC_RELATION_ZERO_SHOT_PROTOCOL_CN.md)
- [源映射兼容性摘要](source_role_mapping_feasibility.json)
- [新CPU检查与实际模型构造](cpu_checks.json)
- [实际源RGB运行见证](runtime_preflight.json)
- [唯一正式队列启动快照](launch.json)
- [初始公开allowlist](public_allowlist.json)

## Limitations

本轮同时改变语义条件与部署读出，没有单模块因果结论。源数学兼容性不是模型准确率；v18混合迁移和v19源拟合失败只提供研究问题。两外域为已查看开发人口，0目标训练不等于新盲测；单seed不作显著性或普适声明。旧模型、原G1／anchor／65轮和负结果保留，原14494／1506／4000及普通／严重名单不变。无收益不自动升最终方法、不延长60轮或追加模块。
