# 配对守恒参考更新：完整匹配对照与方法选择

**最终决定：配对守恒不纳入主方案或论文主贡献。** 两臂均完整65轮、均选best2；完整1506的global NMAE **0.058884298305%FS**，配对守恒 **0.058898092214%FS**。global逐图842较好／660较差／4相同，>1／2／5%FS阈值ID完全相同。配对守恒最差151张mean略低，但已观测完整进程wall高13.1867%；当前匹配证据没有展示实际增量。代码、权重和负结果保留，不扩展模块、loss、seed或轮数。

原G1两臂均10／11、best48与best64保护均5／6，原controller拒绝及原G2未通过均保留。原5%机制筛选的mean项不通过、tail与>5项通过；这一研究筛选不作为整篇论文发表的一票否决。对外评估检查点仍是已冻结best64。

冻结best64已完成Industrial-1395及RF100-151的clean评估，NMAE分别为21.226570%FS与12.691067%FS；完整人口、输入来源和成本见[外部评估报告](../geoprr_v2_external_frozen_20261022/README.md)。这一结果不用于改写匹配机制结论或追加训练。

完整1506、普通1495／1497、3172／2810／3663／9323、全部阈值变化及完整成本见[最终中文报告](final_global_report_cn.md)。独立完整性审计PASS（38／38）。

## 最终公开文件

- [方法选择](final_method_selection.json)：主方案选择、原门槛与资产保留边界。
- [完整summary](final_global_summary.json)与[24组指标CSV](final_global_metrics.csv)：global／preserve／best64／best48及ordinary native／r8在固定四队列上的误差、尾部与阈值统计。
- [global归一化预测](final_global_predictions.csv.gz)与[1506配对行](final_global_paired_rows.csv.gz)：固定ID、group、归一化target／prediction、有效性、误差及队列标志；`absolute_error`是full-scale分数，`*_percent_fs`已乘100。
- [配置](final_global_config.json)、[best验证](final_global_best_validation.json)、[CPU best元数据](final_global_best_metadata.json)、[完成标量](final_global_completed.json)与[65轮history](final_global_history.jsonl)。
- [独立完整性审计](final_global_independent_audit.json)、[论文证据判定](final_global_paper_evidence_decision.json)与[公开allowlist检查](final_global_allowlist_checks.json)。

公开对象按明确字段allowlist构建，不包含原始run.json、H／C／XYZ／像素／几何矩阵、权重或源图像。原始完整资料继续保留private。

## 冻结设计与已有阶段记录

设计与执行边界见[配对守恒设计](../../GEOPRR_V2_REFERENCE_PAIR_PRESERVING_DESIGN_CN.md)、[资源合同](../../GEOPRR_V2_REFERENCE_PAIR_RESOURCE_CONTRACT_CN.md)、[正式匹配合同](../../GEOPRR_V2_REFERENCE_PAIR_UPDATE_PROTOCOL_CN.md)与[论文证据修订](../../GEOPRR_V2_PAPER_EVIDENCE_AMENDMENT_CN.md)。外部评估另按[冻结best64合同](../../GEOPRR_V2_EXTERNAL_FROZEN_EVAL_PROTOCOL_CN.md)执行。

[配对守恒完整候选报告](formal_preserve_report_cn.md)、[候选summary](formal_preserve_summary.json)、[候选完整性审计](formal_preserve_independent_audit.json)、[原controller决定](formal_preserve_original_decision.json)及其全部配置、预测、CPU／资源准入和中期记录继续保留。较早阶段文件记录各自当时的状态，完整匹配结论以本页与`final_global_*`文件为准。

## Limitations

唯一模型seed20262020，源开发1506反复查看；不称跨seed稳定、显著、盲测或外域收益。两臂执行曝光匹配，AMPskip导致有效更新相差2次，所选best2均1418次有效更新。13.1867%来自分开运行的完整进程wall，不是纯算子速度；训练／验证／epoch／进程成本分别保留。
