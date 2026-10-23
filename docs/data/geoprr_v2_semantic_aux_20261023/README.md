# v18完整语义与源结构辅助监督

**完整60轮与source-best58两套外评已完成：源1506 NMAE 0.446599%FS、Industrial1395为13.194496%FS、RF100151为9.929752%FS。** 相对v16，源均值降低15.1302%，Industrial升高12.9081%，RF100均值降低2.3121%，但RF100最差10%均值44.210096→47.466550%FS、max54.413070→68.731612%FS。结果为混合迁移，没有统一收益；不自动替换模型，不延长60轮。

| 模型 | 源NMAE %FS | Industrial %FS | RF100 %FS |
|---|---:|---:|---:|
| v16 source-best59 | 0.526217 | 11.686054 | 10.164776 |
| v18 source-best58 | 0.446599 | 13.194496 | 9.929752 |
| v18 last60 | 0.466742 | — | — |
| 旧GeoPRR历史source-only | — | 11.643618 | 6.607079 |

源RGB人口、路径、物理target与bbox子合同精确等于v16，新增几何子合同独立记录。fresh缓存ImageNet完整MobileNetV4→semantic1280→scalar，共享encoder接受训练期FPN／参考角色辅助监督，固定loss为SmoothL1(beta=.05)+0.1×(roleNLL/(3log256)+fieldL1)。解析标签为源U／M／H及参考角色，P是r=1的虚拟盘面状态；推理只执行direct RGB通路。训练8693701参数／498 state，活动推理8435793／464，辅助257908／34；目标训练／适配／aux调用／反传／更新均0。

源18590 unique、val1506、256 RGB、seed20262020、train64／val64／workers4。无resume；完整60轮后只对冻结source-best58各做一次Industrial1395／RF100151外评。实际1207306图／18925次CNN前向，17453次更新、7次AMP跳过。保存工件审计通过44027项／失败0，最大数值及FP32 loss分解差1.785272263e-9。源best58与last60预测各1506行且不同。

- [完整结果简报](completed/report_cn.md)与[比较表](completed/comparison.csv)
- [完整指标、cohort、focus、成对名单与成本](completed/metrics.json)
- [完整记录](completed/completed_record.json)、[60轮历史](completed/history.jsonl)、[完成审计](completed/numeric_execution_audit.json)
- [完成18文件allowlist](completed/public_allowlist.json)与[本目录allowlist](public_allowlist.json)
- [冻结协议](../../GEOPRR_V2_SEMANTIC_AUX_ZERO_SHOT_PROTOCOL_CN.md)、[启动记录](launch.json)
- [全源解析合同检查](analytic_source_contract.json)、[实际源运行预检](runtime_preflight.json)、[CPU检查](cpu_checks.json)

## Limitations

只有一个源seed；Industrial／RF100是已查看开发集，未提供新盲测、SD或显著性结论，RF100 scalar为四点离线派生。旧GeoPRR两个历史均值和v12 local的RF100 7.659157%FS仍更低。新解析合同未构造Gaussian Q或Q×M×U GT读出，不继承旧Q支持／读数审计保证；一次固定权重实验不足以判断辅助监督的一般效果。初始化关系依赖保存预检，完成审计未加载模型或执行新CNN／旧GT审计，历史像素与完整权重链未重新认证。下一独立[C4轨道语义方案](../../GEOPRR_V2_C4_ORBIT_ZERO_SHOT_PROTOCOL_CN.md)已冻结协议，尚无新性能结果。RGB、权重、原几何、bbox、源完整人口数组、特征与私有合同不公开。
