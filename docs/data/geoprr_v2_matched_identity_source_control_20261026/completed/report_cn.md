# Matched identity15E：没有满足对C3的预注册匹配支持

**科学判定为NO_MATCHED_SUPPORT_FOR_C3。** 同为15E／82440训练呈现／1290 backward尝试，且实际均1289更新＋1AMPskip，C3的两视图R_view和mean／tail drift更低，但两视图>5错误总数59高于identity33、两视图最大误差0.5509077311高于0.5424728990，未满足五项同时成立的预注册判断。Native与CF原有门槛均通过，不等于匹配主判定支持C3；不因source R或漂移较好启动C3长训、promotion或扫描预算／门槛。唯一provided-label保存数字审计171 PASS／0 FAIL，13726数值对比最大差1.4210854715202004e-14；数值接受与标签来源资格见Limitations。

| 预注册匹配条件 | C3 | Identity15E | 对C3支持条件 |
|---|---:|---:|---|
| R_view，normalized full error | 0.06737849594632048 | 0.07481353829650483 | 通过，C3严格更低 |
| 两视图>5%FS错误总数 | 59 | 33 | 未通过，C3不得更多 |
| 两视图最大误差，normalized | 0.5509077310562134 | 0.5424728989601135 | 未通过，C3不得更高 |
| 平均prediction drift，normalized | 0.004645775858379026 | 0.008259031411361233 | 通过，C3严格更低 |
| ceil30 prediction drift，normalized | 0.025007127784192562 | 0.0410818395515283 | 通过，C3严格更低 |

R_view=(native mean＋native ceil30 tail＋CF mean＋CF ceil30 tail)/4，等权平均同source布局的两个固定人工视图。总>5为两视图计数之和，最大误差取两视图最大值；所有字段与主判定使用保存审计结果，不重新选择目标或权重。

| 完整292行条件 | C3 native | Identity native | C3 CF | Identity CF |
|---|---:|---:|---:|---:|
| Mean %FS | 2.583311489 | 2.543121824 | 2.558655712 | 2.635376286 |
| Acc@1 % | 35.273972603 | 34.931506849 | 34.246575342 | 35.616438356 |
| Acc@2 % | 58.904109589 | 62.328767123 | 58.904109589 | 64.726027397 |
| Acc@5 % | 90.410958904 | 94.520547945 | 89.383561644 | 94.178082192 |
| Median %FS | 1.600876451 | 1.549451053 | 1.606027782 | 1.438409835 |
| p90 %FS | 4.834008366 | 4.091511518 | 5.235375166 | 4.164543636 |
| p95 %FS | 6.389944311 | 5.404334664 | 6.280120760 | 5.768549442 |
| ceil30 Tail10 %FS | 11.098201560 | 11.800955331 | 10.711229617 | 12.945961876 |
| Max %FS | 55.090773106 | 53.797036409 | 53.595584631 | 54.247289896 |
| >1总数 | 189 | 190 | 192 | 188 |
| >2总数 | 120 | 110 | 120 | 103 |
| >5总数 | 28 | 16 | 31 | 17 |
| Coverage | 1 | 1 | 1 | 1 |
| 输入／scalar失败 | 0／0 | 0／0 | 0／0 | 0／0 |

每个条件均为73layout×4state、完整292行。同一identity last15的native mean2.543121824／tail11.800955331／max53.797036409%FS，CF2.635376286／12.945961876／54.247289896%FS。C3两个tail更低，identity两个>5计数和两视图最大误差更低，均保留；不把一个有利统计替换预注册主判定。Constant0.5仍用同292行保存标签，完整结果与CSV另存，不作新的训练臂。

| 成对比较 | 图像error改善／恶化／相同 | Layout mean改善／恶化／相同 | Layout max改善／恶化／相同 |
|---|---|---|---|
| C3 native→identity native | 150／140／2 | 38／35／0 | 39／34／0 |
| C3 CF→identity CF | 147／145／0 | 36／37／0 | 36／37／0 |
| identity native→identity CF | 137／152／3 | 33／40／0 | 38／35／0 |

| 比较 | 阈值 %FS | 救回 | 新增 |
|---|---:|---:|---:|
| C3 native→identity native | 1 | 63 | 64 |
| C3 native→identity native | 2 | 64 | 54 |
| C3 native→identity native | 5 | 21 | 9 |
| C3 CF→identity CF | 1 | 67 | 63 |
| C3 CF→identity CF | 2 | 71 | 54 |
| C3 CF→identity CF | 5 | 23 | 9 |
| identity native→identity CF | 1 | 24 | 22 |
| identity native→identity CF | 2 | 23 | 16 |
| identity native→identity CF | 5 | 4 | 5 |

救回为参照>阈值、新条件≤阈值，新增为反向跨越。C3→identity native的>1／2／5救回63／64／21、新增64／54／9；CF救回67／71／23、新增63／54／9。Identity native→CF救回24／23／4、新增22／16／5，CF的>5总数17对native16。两份C3→identity逐图pairs与group CSV保留，全部比较仅用已有scalar行，不重新forward旧C3。

| Prediction绝对漂移 | C3 normalized | Identity normalized |
|---|---:|---:|
| Mean | 0.004645775858379026 | 0.008259031411361233 |
| ceil30 tail | 0.025007127784192562 | 0.0410818395515283 |
| Max | 0.08187556266784668 | 0.30185824632644653 |

Identity最大漂移30.185824633%FS，C3为8.187556267%FS，较大的逐图变化保留。Identity mean／tail drift仍满足原native→CF的0.01／0.05 normalized门槛，失败pair为0；max不是原CF gate阈值。Native／CF原门槛通过与matched主判定否定分开，不表示任一模型完全color invariant。

## 匹配设置与实际曝光

这是一个fresh原生RGB删除对照，C3与identity均固定15E、5496图每E一次、同source packet5496／687parent、native与CF同292／73layout。模型保持完整fresh cached MNV4／conv_head／norm_head／GAP1280／新Linear1，8435793参数／233参数tensor／464state、77BN；RGB512／原photo／ImageNet normalize、SmoothL1 beta.05、AdamW3e-4／wd1e-4／clip5、seed20262020与freshAMP/scaler／batch64／workers4固定。两臂均82440训练呈现／1290back尝试／1289updates＋1skip。

唯一删除是训练通道变化：identity三个nominal schedule slot全部映射RGB，每E实际RGB5496／GBR0／BRG0，15E实际总RGB82440／GBR0／BRG0，每raw图identity15次。History的[1832,1832,1832]仅为nominal slot计数，不代表三颜色各1832；identity直接返回原normalized FP32 tensor，无新增RNG／可学习参数／loss／数据。C3仍为旧冻结C3臂，未重训、延长或补评。

| Epoch | LR | Train SmoothL1 | Train NMAE normalized | Nominal slot0／1／2 | 实际RGB／GBR／BRG | 更新／AMPskip |
|---|---:|---:|---:|---|---|---|
| 1 | 0.0003 | 0.223160355914 | 0.246943711193 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 2 | 0.0003 | 0.133271598345 | 0.155245992468 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 3 | 0.0003 | 0.088485210581 | 0.109772608009 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 4 | 0.0003 | 0.047233113875 | 0.066941564173 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 5 | 0.0003 | 0.030995426493 | 0.049319956587 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 6 | 0.0003 | 0.025893268421 | 0.043446471033 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 7 | 0.0003 | 0.022527793205 | 0.040044913776 | 1832／1832／1832 | 5496／0／0 | 85／1 |
| 8 | 0.0003 | 0.016533606621 | 0.032628962350 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 9 | 0.0003 | 0.013682474381 | 0.029162731407 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 10 | 0.0003 | 0.010833043330 | 0.025320354659 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 11 | 0.0003 | 0.008785553983 | 0.022892645212 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 12 | 0.0003 | 0.010289925972 | 0.024387868579 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 13 | 0.0003 | 0.010381095602 | 0.024707229767 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 14 | 0.0003 | 0.007616548693 | 0.020651467578 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 15 | 0.0003 | 0.007812390842 | 0.021180254642 | 1832／1832／1832 | 5496／0／0 | 86／0 |

前14轮无验证，固定last15、无scheduler／best选择或延长。Native通过constant半mean／ceil30tail学习gate及旧FT5 native guard后，只对同last15同292图作固定RGswap一次；identity native与CF原门槛均通过。Source03、Industrial与RF的RGB／模型／backward／updates均0，没有真实外域结果。

Identity正式83024图／1300 CNN与point调用、1290back尝试；pref另3CNN／8 unique RGB／8 presentations／1combinedback／0optimizer。ID4图的native identity额外decode为CPU工作，计入entry／supervisor wall，与CNN计数分开。三份identity native／constant／CF scalar CSV与73组summary完整保留，另保留C3→identity native／CF的292行pairs及group summaries。

| 保存成本边界 | 秒 |
|---|---:|
| 15E train保存和 | 446.680151045322 |
| Native＋CF pipeline保存和 | 6.870805330575 |
| Formal child process | 458.994210168719 |
| Formal matched-wrapper full entry | 460.987576279789 |
| Pref full entry | 18.396342489868 |
| Supervisor pref stage | 19.270640596747 |
| Supervisor control stage | 462.370882876217 |
| Whole supervisor | 482.665441177785 |
| Individual point／isolated operator time | unknown／null |

Formal peak allocated／reserved8262538752／11345592320bytes；pref630110208／725614592bytes单列，完整phase time／peak保留在[成本](costs.json)。Pref18:38:15.954471→18:38:35.224071 UTC，control18:38:35.345544→18:46:17.715273 UTC，两stage complete／exit0；父424199、pref424202、control424240退出，18:52:49保存观察GPU空。

科学执行版本3e39174f36b342691f6ee96acdea2036ee5e8d30；C3参考科学版本45de72cbd7f6c0ac1e507f0e357760e77e2a3933、归档版本78ae50613848d313063ae554ea0d54aa429daec8。仅规范化保存scalar/pairs/groups/summaries/history/gates/matched comparison/config/run/completed/queue/screen/cost/init及source-isolation／resume metadata公开，完整合同、训练arrays、RGB、weights、features、geometry和privatepaths保持私有。

## Limitations

本次最终审计为171 PASS／0 FAIL、13726 numeric comparisons，最大绝对差1.4210854715202004e-14。参考统计按既有绝对容差1e-12接受末位roundoff，并非逐位一致：C3 native Acc@5的冻结值90.41095890410959与保存参考90.41095890410958差1.4210854715202004e-14，bitwise_equal=false。公开只保留这一正确数值接受状态、容差和边界。

这是provided-label保存数字核对，不新增raw-label派生认证。旧C3审计仍为149 checks／148 PASS／1 FAIL，缺normalized_progress／y字段的覆盖限制与未观测到label数值冲突的结论保留，旧档案不变；本次native／CF targets与已有C3／directFT5保存targets对齐，不意味着physical label归一化全链已独立认证。见[参考资格](reference_audit_qualification.json)及[旧C3审计](../../geoprr_v2_cyclic_color_source_control_20261026/completed/numeric_execution_audit.json)。

一seed／73source layout×4state，native与RGswap是相关人工视图，不是独立新域／新盲测，不支持SD、显著性或真实Industrial／RF泛化。虽然scheduled和actual更新预算相同，初始化相同的证据仅factory／cache／seed／冻结代码声明，未打开旧权重内容。cached462、strict464、BN77源+1／eval+0、233参数0opt不变、梯度／dtype等为保存见证，不是本次新模型或checkpoint内容实验。

Matched wrapper entry在torch import前开始，已保存entry/full与supervisor wall包含wrapper metadata／reference检查；child full排除最终completion JSON和downtime。独立point与operator compute未测为null，不能把wall差直接当算子净开销；pref额外4ID decode及wrapper检查包含在相应wall。

预注册五条件未同时满足，结论保持NO_MATCHED_SUPPORT_FOR_C3；不能只用较低R_view或漂移启动C3长训，也不把本结论扩大为所有C3／所有颜色随机化无效。不promotion、不新seed／order／预算／门槛扫描，不自动延长、source03或真实外域评估，全部旧正负结果保持。
