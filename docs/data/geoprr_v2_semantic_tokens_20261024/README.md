# v23全语义与多尺度空间latent tokens

**独立表示／读出与源风险选择规则已冻结，尚无新模型性能或启动结论；实际参数与state数待CPU构造后记录。** 单次完整MobileNetV4前向取得semantic1280及四尺度空间特征，stride4先pixel-unshuffle保留有序2×2 pattern，四级投影形成2368个128维memory token。4个semantic条件通用query经两层独立标准Transformer decoder，最后LN／flatten512→128→1与float sigmoid输出唯一reading。全部CNN与读出端到端训练，没有旧point残差、几何角色头、融合router或teacher。

数据完全复用v22：22686 unique，旧18590一次＋roll4096各4次＝34974曝光／547批，源验证1506＋512＝2018／32调用，无新渲染、名单或增强改变。唯一训练loss为source SmoothL1 beta.05。四个query没有实体针尖／量程start／end认证，位置只来自规则ROI cell；模型只收RGB，不输入GT几何。

预先固定R=(mean_original＋mean_roll＋tail10_original＋tail10_roll)／4，tail复用完整误差top ceil10%（151／52行，失败error1包含）。同一R用于最早strict best、LR plateau及20轮patience；min60／max160，不强凑旧114轮，也不根据目标挑权重。新结构与source-tail选择一起改变，不作纯结构单因素解释。

source结束后冻结best，各一次Industrial1395／52与RF100151／35；目标不训练、拟合统计、适配BN或选择checkpoint／epoch／LR／阈值／ROI／温度。E轮正式图数36992E＋1546、CNN调用579E＋25，updates＋AMP skip=547E；decoder额外计算／时间／显存另记，不报计划已完成。新结构实际见证计划3CNN／8图／1back／0opt，不重跑旧GT／预检；全部完成后再通知完整结果。

- [冻结协议](../../GEOPRR_V2_SEMANTIC_TOKENS_ZERO_SHOT_PROTOCOL_CN.md)
- [实际训练器CONFIG对应配置](config.json)
- [初始公开allowlist](public_allowlist.json)
- [复用的v22源完成记录](../geoprr_v2_roll_source_20261024/source_completed.json)
- [v22完整结果](../geoprr_v2_roll_source_20261024/completed/report_cn.md)

## Limitations

目标是mean与严重尾部同时改善，不能只拿Acc收益替代；源R也不保证外域改善。roll512共享HDR、精确0／1人数0，20096原metadata无seed，非全历史seed认证或新域盲测。两外域已查看、单seed不称显著性；常规attention与论文动机不是本任务有效证据。原14494／1506／4000、普通／严重名单和全部旧模型／失败保留，不预先promotion或机械延长。

实际构造为9374417参数／293个parameter tensor／528 state，其中新读出939905参数。7项新CPU toy检查通过；实际新结构AMP运行见证3次CNN／8图／1次反传／0更新通过，完整cached encoder462 state保留、四尺度与2368memory／4query路径及必要梯度通过、77 BN source+1／eval0、528 state严格加载。当前独立源训练已启动，外域尚未评估；见[cpu_checks.json](cpu_checks.json)、[runtime_preflight.json](runtime_preflight.json)与[launch.json](launch.json)。这些是运行证据，不是泛化性能结果。
