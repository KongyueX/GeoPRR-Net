# Mean-tail source risk control

状态：新风险聚合零更新源预检通过，固定15E训练进行中，终点source-ID/条件CF尚未评估。科学743acb16b486f1f494c3b1f49fa7c4a41848887d已冻结推existingmain。

原模型/原生RGB/source数据/seed/15E预算不变，只将source SmoothL1聚合为.5mean+.5batch topceil10%。原生与固定CF都须满足mean/tail/max/>5及准确参考样本新大错联合guard，不能只tail好通过。source_training_smooth_l1历史值为composite risk，非primitive mean；trainNMAE非val。两个real域与needle03全0RGB/model。详见[协议](../../GEOPRR_V2_MEAN_TAIL_SOURCE_CONTROL_CN.md)、[机器协议](protocol.json)、[启动](launch.json)、[运行见证](runtime_preflight.json)。明确allowlist metadata之外的raw合同/RGB/weights/features留server。
