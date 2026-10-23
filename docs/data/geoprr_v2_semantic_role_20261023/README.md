# v21完整语义直接条件角色能量

**60轮与两套冻结外评完整结束；source-best42源NMAE为0.258947%FS，Industrial为23.978974%FS，RF100为14.008518%FS。** 相对匹配base v20，均值分别降低8.6007%／18.0417%／13.6561%，这是保留的连续增量；两外域最差10%均值升高，且仍弱于v16直接读数和旧GeoPRR历史参照，因此不作为默认／最终模型、不延长训练。全部资产及结果保留。

完整fresh v20的502项base state之后增加全零、无bias的Linear1280→3×16，新增61440参数／1 state；query与原角色token点积后加入unary，原local共享A联合和全质量读出保持。完整8823492参数／503 state／263个参数tensor，模型只接收RGB，源H／span／U／M／虚拟角色只进loss。每图固定权重query不是目标适配，P不是实体针尖。

只用18590源训练／原1506验证、原256 RGB处理与seed20262020，全部60结束后才各一次冻结source-best42外评；目标训练／统计拟合／BN适配／checkpoint及超参数选择均0。源best／last和两外域input／scalar全部有效，coverage1。源ordinary1495均值改善，但原严重6均值0.822225→11.047740%FS；best42最大45.139897%FS（sync_338），last60最大55.712643%FS（sync_4658），完整错误交换照报。

实际17460次反传尝试／17453次更新／7次AMP跳过，无resume，正式1207306 CNN图／18925调用；query／关系头／读出训练17460、验证1440、目标25次，目标反传／更新0。一次数值／执行核对54364项通过、0失败。完整逐图、组、fixed cohort、阈值救回／新增、尾部及成本已归档。

- [完整报告](completed/report_cn.md)
- [完整指标](completed/metrics.json)
- [模型比较](completed/comparison.csv)
- [实际成本](completed/costs.json)
- [审计](completed/numeric_execution_audit.json)
- [冻结协议](../../GEOPRR_V2_SEMANTIC_ROLE_ZERO_SHOT_PROTOCOL_CN.md)
- [CPU检查](cpu_checks.json)
- [实际源运行见证](runtime_preflight.json)
- [启动记录](launch.json)
- [公开allowlist](public_allowlist.json)

## Limitations

两域均值改善不等于尾部或总体最佳，单seed与已查看开发人口不称新盲测或显著性。初始reading／分布在1e-6容差内匹配，不是普适逐位保证；相同base初始化不等于后续随机轨迹相同。outside_arc_mass不能硬门控正确性。旧模型、原G1／anchor／65轮和负结果保留，原14494／1506／4000及普通／严重名单不变。RGB／权重／原始几何与合同保持私有。
