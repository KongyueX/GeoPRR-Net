# v22真实相机roll源覆盖：完整结果

**源训练按预先规则在114轮结束，最早源score最优为best94；冻结外评Industrial NMAE为12.691571%FS、RF100为10.867316%FS。** 相对v16两域均值分别高8.6044%／6.9115%，最差10%均值也更差，因此不替换默认模型、不延长本配方。两域Acc@1／2／5均提升，保留主体小误差准确率与严重尾部的真实取舍。

Industrial Acc@5从44.874552%升至52.329749%，RF100从61.589404%升至65.562914%；最差10%均值同时从45.734073／44.210096升至55.381639／52.215675%FS。相对v18为Industrial均值更好、RF更差，相对v21两域均值更好；旧GeoPRR历史11.643618／6.607079%FS仍更低，不能混成统一提升。

best94原1506源均值0.489733%FS、新roll512为0.671073%FS，等权score为0.580403%FS；last114分别0.514028／0.694045、score0.604037，last未外评。原源最大sync_615为92.611902%FS，仍保留在原普通1495／1497；没有删样本或重分人口。所有源／外域input和scalar有效、coverage1，新rollval精确0／1人数均0，不假称端点评估保证。

模型为完整fresh缓存ImageNet MobileNetV4 semantic1280→point，单图一次CNN、8435793参数／464 state。旧18590保留，追加4096真实roll训练图并各重复4次；每轮34974曝光／547批，unique22686。新512源验证仅新布局／ID／group／generated seed独立，共享HDR，原1506旧scene guard另保留。只用源等权score选best及控制LR／停止；两外域没有参与训练、统计拟合、BN适配或选择。

实际训练3987036图／62358尝试＝62337更新＋21 AMP跳过，验证230052图／3648调用，外域1546／25／0back／0update；正式总4218634图／66031 CNN，no resume。源生成、零更新预检和正式模型成本分开。一次独立CPU保存工件核对73060项通过、0失败，最大差1.421085472e-14；完整源best／last四份逐图、两外域逐图／组／summary、114轮history与成本已归档。

- [完成报告](completed/report_cn.md)
- [完整指标](completed/metrics.json)
- [模型与准确率比较](completed/comparison.csv)
- [实际成本](completed/costs.json)
- [审计](completed/numeric_execution_audit.json)
- [114轮history](completed/history.jsonl)
- [best双源验证](completed/best_validation.json)
- [last双源验证](completed/latest_validation.json)
- [冻结协议](../../GEOPRR_V2_ROLL_SOURCE_ZERO_SHOT_PROTOCOL_CN.md)
- [配置](config.json)
- [源生成完成](source_completed.json)
- [零更新运行见证](runtime_preflight.json)
- [公开allowlist](public_allowlist.json)

## Limitations

本轮性能优先，没有等预算对照要求；新增数据／重复曝光／源验证和adaptive学习共同改变，不能单因素归因roll。新rollval不是背景／域盲测，没有旧v16在该新人口上的推理基线；旧历史像素／完整权重链未重新认证。两外域已查看、单seed不称显著性或普适机制。原14494／1506／4000与全部旧模型、G1／anchor／65轮失败保留；RGB／权重／原始几何、完整源数组及合同保持私有，不自动promotion、不延长当前配方。
