# v23全语义与多尺度latent tokens：完整结果

**117轮源训练及两套冻结外评已完成，source-R最早最优为best97；Industrial NMAE为21.426597%FS，RF100为13.046721%FS。** 两域mean、最差10%及max均弱于v16和旧GeoPRR历史参照，未达到整体泛化与严重尾部目标，不自动升模型、不延长。RF100最差10%相对v22下降，但均值与max升高，局部取舍保留，不能写成统一收益或所有attention无效。

best97原1506 mean／tail为0.490210／2.440468%FS，roll512为0.740396／3.029107%FS，四项等权R=1.675045%FS。last117 R=1.975943%FS；best97之后20完整轮无严格新R，按预声明source patience停止。全部source后才各一次锁定best97外评，last仅源记录；两外域无训练、统计／BN适配或选择，全部input／scalar有效、coverage1。

实际9374417参数／293个parameter tensor／528 state，单CNN完整semantic1280加四尺度memory2368×128、四通用query与两独立decoder，全部端到端训练。没有旧point残差、GT角色、几何router或teacher，规则cell位置不是GT针尖。源数据与256 RGB处理完全复用v22，无新增渲染；结构与mean-tail source selector一起改变，不作纯attention因果解释。

正式4091958训练图／63999反传尝试＝63974更新＋25 AMP跳过，验证236106／3744调用，外域1546／25／0back／0update，总4329610图／67768 CNN，无resume。一次独立CPU保存工件核对73780项通过、0失败；四份源best／last、两外域逐图／组、117轮history与全部成本归档。

- [完整报告](completed/report_cn.md)
- [完整指标](completed/metrics.json)
- [模型与分位／准确率比较](completed/comparison.csv)
- [实际成本](completed/costs.json)
- [审计](completed/numeric_execution_audit.json)
- [117轮history](completed/history.jsonl)
- [best源R分量](completed/best_validation.json)
- [last源R分量](completed/latest_validation.json)
- [冻结协议](../../GEOPRR_V2_SEMANTIC_TOKENS_ZERO_SHOT_PROTOCOL_CN.md)
- [配置](config.json)
- [实际模型预检](runtime_preflight.json)
- [公开allowlist](public_allowlist.json)

## Limitations

单seed、已查看两外域不构成显著性或新盲测；roll512只在布局／ID／group／声明seed层面独立且共享HDR，端点人数0，历史metadata seed链未认证。原fixed cohorts、615及其他大错均保留，不删行重分人口；旧历史模型像素／完整权重链未重新认证。全部旧模型与负结果留存，不能用局部Acc或单个case替代整体mean／tail目标，原始RGB／权重／几何／特征和完整合同继续私有。
