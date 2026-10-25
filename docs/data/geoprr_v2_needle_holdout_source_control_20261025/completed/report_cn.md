# 五轮源针型留出控制：均值通过，尾部门槛未通过

**在固定五轮预算内，模型已改善source-ID平均读数，但尾部未满足预先声明的双门槛；控制正常完成为learning_gate_failed，留出针型问题没有被评估。** 同一292行source-ID上，last5模型mean6.132800788%FS低于门槛12.504463703%FS，tail10为25.669996335%FS、高于门槛23.692208053%FS。因此按冻结协议跳过source-OOD，不扩预算、不改阈值、不外评。唯一独立保存标量／执行核对49／49通过、0失败，总结为PASS_WITH_QUALIFICATIONS；科学结果与审计结论分别保留。

| 预先声明的门槛 | Constant0.5 %FS | 模型last5 %FS | 模型允许上限 %FS | 判定 |
|---|---:|---:|---:|---|
| Mean | 25.008927406 | 6.132800788 | 12.504463703 | 通过 |
| Tail10，ceil30 | 47.384416106 | 25.669996335 | 23.692208053 | 未通过 |

门槛同时要求model mean和tail10各≤constant0.5的一半，固定292分母、tail30。保存的mean／tail比值分别为0.2452244627746543／0.5417392139555451；总体gate为false。均值收益成立，尾部虽比constant更低仍未达到一半要求。

| Source-ID结果 | Last5模型 | Constant0.5 |
|---|---:|---:|
| Mean %FS | 6.132800788 | 25.008927406 |
| Acc@1 % | 16.095890411 | 2.397260274 |
| Acc@2 % | 28.424657534 | 4.794520548 |
| Acc@5 % | 60.958904110 | 8.219178082 |
| Median %FS | 3.886425029 | 24.896581471 |
| p90 %FS | 12.683204412 | 44.168485999 |
| p95 %FS | 17.504554670 | 47.623561192 |
| Tail10 %FS | 25.669996335 | 47.384416106 |
| Max %FS | 57.870632410 | 49.942222197 |
| Acc@1正确数／分母 | 47／292 | 7／292 |
| Acc@2正确数／分母 | 83／292 | 14／292 |
| Acc@5正确数／分母 | 178／292 | 24／292 |
| >1／2／5总数 | 245／209／114 | 285／278／268 |
| Coverage | 1 | 1 |
| 输入／scalar失败 | 0／0 | 0／0 |

Model与constant使用同一顺序、group和既有scalar标签的完整292行。合法0／1预测仍为有效输出；本次保存的边界饱和预测数均0。模型max57.870632410%FS高于constant的49.942222197%FS，尾部代价与均值收益同时报告。完整[模型292行](source_id_model_predictions.csv.gz)、[constant292行](source_id_constant_predictions.csv.gz)、两套73组CSV与summary均保留，不生成未评估人口的预测。

| 预先声明的metadata分层 | N／layout | Model mean %FS | Constant mean %FS | Model tail10 %FS | Constant tail10 %FS |
|---|---|---:|---:|---:|---:|
| canonical_pointer_01 | 124／31 | 5.279111304 | 25.080881608 | 21.878395745 | 46.459242336 |
| canonical_pointer_02 | 168／42 | 6.762904930 | 24.955818352 | 28.132210858 | 47.985113853 |
| state-slot 0 | 73／73 | 4.971382910 | 37.826616462 | 20.566676720 | 48.887834413 |
| state-slot 1 | 73／73 | 4.683258395 | 11.102995970 | 15.975209512 | 22.108871117 |
| state-slot 2 | 73／73 | 7.254412546 | 13.864238621 | 24.893021770 | 23.152723908 |
| state-slot 3 | 73／73 | 7.622149301 | 37.241858571 | 35.045402497 | 48.373409361 |

分层仅描述冻结metadata，未添加分组门槛或依分层选择模型。各分层tail按其已有N分别取ceil10%；全体gate仍只使用292行／tail30。

| 人口／阶段 | 声明图数 | 实际模型调用 | 状态／指标 |
|---|---:|---|---|
| Train，两种已见针型 | 5496 unique／每轮一次 | 5轮／430 CNN调用 | 完整结束 |
| Source-ID | 292／73layout | last5一次／5 CNN调用 | 上表完整结果 |
| Source-OOD，留出针型 | 220／55layout | RGB0／CNN0／point0 | skipped_learning_gate_failed；metrics=null |
| Industrial／RF | 不纳入本控制 | images／CNN／point／backward／updates均0 | 无结果 |

这是一项独立的有界源问题定位控制，不是正式V32或泛化候选。训练两种针型、留出canonical_pointer_03由source metadata字典序预先决定，未用预测或真实外域选择。训练为687个完整parent，2748 original＋2748 roll=5496 unique；每layout四state闭包、train／ID／OOD的sample与layout分离来自冻结metadata和服务端合同相等证据，不是新像素／GT核实。旧14494无因素记录训练行、原1506验证及留出种2696训练图未纳入，不改历史split或分母。

| Epoch | 固定LR | Train SmoothL1 | Train NMAE normalized | 图数／CNN调用 | 更新／AMPskip |
|---|---:|---:|---:|---|---|
| 1 | 0.0003 | 0.223160355914 | 0.246943711193 | 5496／86 | 86／0 |
| 2 | 0.0003 | 0.133271598345 | 0.155245992468 | 5496／86 | 86／0 |
| 3 | 0.0003 | 0.088485210581 | 0.109772608009 | 5496／86 | 86／0 |
| 4 | 0.0003 | 0.047233113875 | 0.066941564173 | 5496／86 | 86／0 |
| 5 | 0.0003 | 0.030995426493 | 0.049319956587 | 5496／86 | 86／0 |

每轮86批，batch64、末56，五轮27480图／430次CNN与backward尝试＝430更新＋0AMPskip。前四轮不验证、训练内validation calls均0；第5轮结束后只对source-ID做一次292图／5调用，正式总27772图／435 CNN调用，point逐阶段相同。只保留last5，无best选择、完整V26源R、scheduler、早停或延长。source-OOD未读取RGB或前向，heldout针型表现保持未评估。

表示保持完整fresh cached ImageNet MobileNetV4／GAP1／conv_head／norm_head／semantic1280／新Linear1280→1 std.001 bias0／FP32sigmoid，8435793个trainable参数／233参数tensor／464state。全部CNN与point源训练，77BN仅源训练更新、eval冻结；RGB512／output128、native ROI／LINEAR／ImageNet normalize／原photo、唯一SmoothL1(beta=.05)、fresh AdamW3e-4／wd1e-4／clip5／seed20262020与原AMP固定。没有GN、style、二阶head、teacher、几何分支、融合、TTA、新权重或旧task checkpoint。

Source-ID-only pref单列3CNN／8图／1combinedback／0optimizer，4训练图＋两次各2张source-ID，source-OOD RGB／forward0。cached462初始化保持、233参数零optimizer不变、77BN源+1／eval额外0、stem／neck／point必要梯度有限非零、464 strictreload和source-ID pipeline identity均为保存见证及执行metadata一致性；独立审计未fresh打开checkpoint或cached权重内容。正式fresh初始化和default GradScaler，不沿用pref状态。

| 实际成本与边界 | 秒 |
|---|---:|
| 正式source_control supervisor stage wall | 173.437898 |
| Pref supervisor stage wall，单列 | 17.985979 |
| 两stage wall之和 | 191.423877 |
| Pref起点至control结束 | 191.527038 |
| 五轮training保存和 | 163.561800412834 |
| 最终source-ID pipeline | 4.893382981420 |
| 最终source-ID model forward | 3.400220595300 |
| 原helper process_seconds，带边界qualification | 171.235725361854 |
| 原helper recorded full_process_seconds，带边界qualification | 171.392929736525 |
| Point独立计时 | unknown／null |

**成本边界qualification：** 新control先lazy import torch，再导入设置计时起点的旧runtime helper。因此保留的recorded full_process_seconds及其旧边界标签不能代表新entry完整import；实际从helper module import（torch已导入）计到runtime metadata求值、最终completed.json写入之前，包括训练、source-ID、scalar保存及最终last.pt保存。完整child-stage elapsed采用supervisor launch前started_at到wait后finished_at的173.437898秒；原记录不改。pref也以独立supervisor wall17.985979秒报告。point未单独测量，不填0；其计算包含在完整model forward／pipeline内。

Source peak allocated／reserved8262538240／11345592320bytes；pref peak630110208／725614592bytes单列，常数baseline不运行CNN／point，其标量处理不被描述为零成本。完整[成本与qualification](costs.json)保留所有原helper值、修正后的真实边界和不同wall边界，不将它们混作一个总时间。

Pref于2026-10-25T13:47:20.021162+00:00启动、13:47:38.007141结束；正式control于13:47:38.110302启动、13:50:31.548200结束，两stage complete／exit0。13:52:29.672489保存观察中父进程413065、pref413069和control413106已退出，GPU空闲。gate未通过属于正常控制结果。

[独立审计](numeric_execution_audit.json)保留49／49 PASS、0失败及PASS_WITH_QUALIFICATIONS，限定为保存标量与执行metadata核对；保留cost边界、保存见证／内容未fresh核实、metadata闭包范围和早期root收集范围四项qualification。本次公开只含规范化保存标量／metadata，数据最小化陈述仅针对本次公开工件；不将独立收集的服务端剔除范围扩写成整个root历史收集过程的声明。

## Limitations

本结果只支持固定五轮、两种已见合成针型的source-ID平均学习已改善，但预声明尾部条件未满足。留出针型没有被评估，不能推断其表现、针型是外域错误成因、真实域收益、另一预算下的可学性、显著性或普适性。source-ID与OOD其他layout因素可能混杂，不建立font／HDR新域结论。

审计只核对既有scalar和冻结metadata，不构造模型、不运行GPU／forward／新GT或像素／mesh核实。saved reload、cached462／233／464／BN77与保存RNG机制只保留保存见证和metadata一致性，不写成新内容重载验证。完整合同、source arrays、5496训练ID名单、RGB、weights、features、geometry、私有路径与commands不公开。全部旧资产保留，不自动promote；固定5E预算、门槛和条件执行保持，不扩轮数、不换阈值、不补外评。
