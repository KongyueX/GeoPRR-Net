# v17完整语义与空间残差读出

独立结构实验协议已冻结，新CPU和真实源RGB运行检查通过，正式训练尚未启动。本目录暂不包含本轮训练性能。

完整冻结v16 source-best59，保留原语义和标量通路，在同一次CNN前向的stride8局部特征上增加3个通用空间token和零初始化logit残差。仅新增129057可训练参数；完整模型8564850参数／474 state。原父模型464 state及77个BN counter逐项保持，初始预测与父模型精确相同。

固定8完整源epoch，原18590训练／1506验证、256 RGB输入，唯一scalar SmoothL1 beta0.05。只按源验证在父模型epoch0与8轮新checkpoint中选择。完成后才各做Industrial1395／RF100151冻结外评；若选中epoch0则复用原外评，不重复CNN。

4项新CPU检查通过（0.722秒）。真实源运行检查使用4张unique端点RGB，2次CNN／8图前向／2次branch反传／0次CNN反传／0optimizer；初始identity、attention有限归一化、全部父state及BN不变、临时非零输出层的上游梯度和474 state严格reload通过。临时分支变化已全部恢复，正式训练重新初始化，不使用检查后的状态。

- [冻结协议](../../GEOPRR_V2_SEMANTIC_SPATIAL_ZERO_SHOT_PROTOCOL_CN.md)
- [实际源运行检查](runtime_preflight.json)
- [新CPU检查](cpu_checks.json)
- [v16已完成结果](../geoprr_v2_semantic_direct_20261023/completed/report_cn.md)

## Limitations

本目录是实现与输入检查，不是新性能。3个空间token没有被认证为针尖或刻度角色，局部特征是否改善zero-shot仍须实测。Industrial与RF100已查看且仅作冻结开发评估，0目标训练／统计拟合／BN适配／checkpoint选择，不称新盲测。全部旧模型与失败保留，私有像素、权重和原几何不公开。
