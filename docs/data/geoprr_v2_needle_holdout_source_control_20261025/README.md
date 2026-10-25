# 五轮源针型留出控制

**固定五轮内source-ID平均学习已改善，尾部未满足预先声明的双门槛；控制正常完成为learning_gate_failed，source-OOD按协议跳过。** 同一292行上，模型mean6.132800788%FS≤门槛12.504463703%FS；tail10为25.669996335%FS>门槛23.692208053%FS。constant0.5的mean／tail为25.008927406%FS／47.384416106%FS。均值通过、尾部未通过，heldout针型问题保持未评估；这不是正式V32或泛化候选。

[完整报告](completed/report_cn.md)、[指标](completed/metrics.json)、[学习门槛](completed/learning_gate.json)及[独立审计](completed/numeric_execution_audit.json)保留49／49 PASS、0失败、PASS_WITH_QUALIFICATIONS和科学结果。模型Acc@1／2／5为16.095890／28.424658／60.958904%，constant为2.397260／4.794521／8.219178%；模型max57.870632%FS高于constant49.942222%FS，不能只报均值收益。完整model292／constant292 scalar CSV及各73组CSV／summary保留。

[冻结协议](../../GEOPRR_V2_NEEDLE_HOLDOUT_SOURCE_CONTROL_CN.md)只用两种source metadata预选针型的5496既有RGB（2748 original＋2748 roll，每轮各一次），687个完整parent；source-ID292／73layout，声明source-OOD220／55layout。表示保持V26完整MNV4／GAP1280／单scalar读出，fresh cached ImageNet初始化、新point、全CNN／point源训练、77BN源更新／eval冻结；8435793参数／233参数tensor／464state，RGB512／原photo与固定LR3e-4／SmoothL1。22项NEWCPU检查及source metadata准备完成，不重复执行。

正式仅last5，5轮27480图／430 CNN与backward尝试＝430更新＋0AMPskip，前四轮不验证；最后source-ID一次292图／5调用，正式总27772图／435 CNN与point调用。source-ID-only pref3CNN／8图／1back／0optimizer单列，OOD RGB／forward均0。[source-OOD状态](completed/source_ood_status.json)为skipped_learning_gate_failed、metrics null；不生成假预测。Industrial／RF的images／CNN／point／backward／updates全0，没有外域结果或baseline比较。

**成本qualification保留：** 旧helper记录process171.235725362秒、recorded full171.392929737秒，但新entry先lazy import torch再导入计时helper，因此旧full边界标签不能代表新entry完整import。完整正式child-stage wall用173.437898秒，pref17.985979秒单列；两stage和191.423877秒，pref起点至control结束191.527038秒。Point独立计时unknown／null，原helper值不改，真实边界及peak详见[成本](completed/costs.json)。cached462／233参数零更新保持／464 strictreload／BN77仅为保存见证与执行metadata一致性，不是fresh checkpoint内容核实。

根7项入口＋[completed21项](completed/public_allowlist.json)，[总显式清单](public_allowlist.json)28项。本次公开只含规范化保存标量／metadata；full source合同、source arrays、5496训练ID名单、RGB、weights、geometry、features、privatepaths与commands不公开。run／completed／queue／screen明确为metadata projection，四项audit qualification完整保留。原始记录及全部历史资产保留，不扩5E、不改门槛、不补外评、不自动promote。
