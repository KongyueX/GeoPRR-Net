# C3颜色通道源控制：科学门槛通过，唯一审计保留FAIL

**保存结果的科学判定为SOURCE_COLOR_CONTROL_PASSED；唯一fresh审计状态仍为FAIL：149项中148 PASS、1 FAIL。** Native mean2.583311489／tail11.098201560／max55.090773106%FS，固定RGswap counterfactual（CF）为2.558655712／10.711229617／53.595584631%FS；预声明native与CF门槛均满足。归档保留有资格限定的保存数字证据，不作模型promotion；审计唯一失败的具体边界与字段存在性证据集中见Limitations。

| 完整292行条件 | 旧directFT5，5E | Native，15E last15 | Constant0.5 | CF，固定RGswap last15 |
|---|---:|---:|---:|---:|
| Mean %FS | 6.132800788 | 2.583311489 | 25.008927406 | 2.558655712 |
| Acc@1 % | 16.095890411 | 35.273972603 | 2.397260274 | 34.246575342 |
| Acc@2 % | 28.424657534 | 58.904109589 | 4.794520548 | 58.904109589 |
| Acc@5 % | 60.958904110 | 90.410958904 | 8.219178082 | 89.383561644 |
| Median %FS | 3.886425029 | 1.600876451 | 24.896581471 | 1.606027782 |
| p90 %FS | 12.683204412 | 4.834008366 | 44.168485999 | 5.235375166 |
| p95 %FS | 17.504554670 | 6.389944311 | 47.623561192 | 6.280120760 |
| ceil30 Tail10 %FS | 25.669996335 | 11.098201560 | 47.384416106 | 10.711229617 |
| Max %FS | 57.870632410 | 55.090773106 | 49.942222197 | 53.595584631 |
| >1总数 | 245 | 189 | 285 | 192 |
| >2总数 | 209 | 120 | 278 | 120 |
| >5总数 | 114 | 28 | 268 | 31 |
| Coverage | 1 | 1 | 1 | 1 |
| 输入／scalar失败 | 0／0 | 0／0 | 0／0 | 0／0 |

Native、constant和CF均为73个layout×4state的完整292行；旧directFT5只使用既有同292行保存输出，不补训或补评。Native max55.090773106%FS和CF max53.595584631%FS仍大；CF mean／tail略低，但Acc@1／5更低、p90更高及新增大错同时保留。完整三份scalar CSV、各73组CSV与summary可直接查看，失败按完整分母error1保留，合法饱和值不误判invalid。

| 预声明判断 | 保存结果 | 判定 |
|---|---|---|
| Native原constant双学习门槛 | mean2.583311489≤12.504463703%FS；tail11.098201560≤23.692208053%FS | 通过 |
| Native相对旧directFT5 primary | mean、tail、max和>1／2／5计数全部满足固定上限 | 通过 |
| CF进入条件 | Native learning且native primary均通过，同一last15 | 满足 |
| CF读数门槛 | mean2.558655712≤12.504463703%FS；tail10.711229617≤23.692208053%FS | 通过 |
| CF prediction drift门槛 | mean0.464577586≤1%FS；ceil30 tail2.500712778≤5%FS | 通过 |
| 保存科学判定 | SOURCE_COLOR_CONTROL_PASSED | source控制门槛通过 |

| 成对比较，同292行 | 图像error改善／恶化／相同 | Layout mean改善／恶化／相同 | Layout max改善／恶化／相同 |
|---|---|---|---|
| 旧directFT5→native | 215／77／0 | 68／5／0 | 66／7／0 |
| native→CF | 154／133／5 | 40／33／0 | 39／32／2 |

| 比较 | 阈值 %FS | 救回 | 新增 |
|---|---:|---:|---:|
| 旧directFT5→native | 1 | 86 | 30 |
| 旧directFT5→native | 2 | 123 | 34 |
| 旧directFT5→native | 5 | 97 | 11 |
| native→CF | 1 | 9 | 12 |
| native→CF | 2 | 11 | 11 |
| native→CF | 5 | 4 | 7 |

救回为参照>阈值、新条件≤阈值，新增为反向跨越。旧→native >1／2／5救回86／123／97、新增30／34／11；native→CF救回9／11／4、新增12／11／7。CF >5总数31对native28，净新增3，不能被mean／tail略低掩盖。

| CF相对native的prediction绝对漂移 | %FS | 保存normalized值 |
|---|---:|---:|
| Mean | 0.464577586 | 0.004645775858379026 |
| ceil30 Tail | 2.500712778 | 0.025007127784192562 |
| Max | 8.187556267 | 0.08187556266784668 |

保存failed pairs为0。漂移是同图CF prediction与native prediction之差，和相对scalar标签的reading error分别保存；仍有最大8.187556267%FS漂移与较大读数错误，不作完全color invariance结论。

## 固定控制与执行

完全保留fresh cached ImageNet完整MNV4 conv-medium／conv_head／norm_head／GAP1／semantic1280／新Linear1280→1 std.001 bias0／FP32sigmoid，8435793参数／233参数tensor／464states、77BN；全部CNN与point源训练，BN源更新／eval冻结。输入仍nativeROI→LINEAR512／ImageNet，output128仅API、geometryFalse／pairNone、原photo。训练复用5496图／687parent、两seen针型；source03、Industrial、RF全部0 RGB／模型调用，没有真实外域结果。

训练只用RGB／GBR／BRG三个C3元素，order编号为(dataset_index+epoch−1)%3，原photo后、encoder前在FP32 normalized RGB上作固定仿射通道置换，等价于raw RGB换通道后使用同一ImageNet归一化。每轮5496呈现，三order各1832；固定15轮，每raw item每order5次。原生ID使用identity，预留CF仅固定[1,0,2] RGswap，在训练C3之外，同一last15只评同292图一次。无额外RNG、clip、grayscale、resize、feature cache、RGB写盘、可学习算子参数或额外loss。

| Epoch | LR | Train SmoothL1 | Train NMAE normalized | Order0／1／2曝光 | 更新／AMPskip |
|---|---:|---:|---:|---|---|
| 1 | 0.0003 | 0.216157110281 | 0.239794608113 | 1832／1832／1832 | 86／0 |
| 2 | 0.0003 | 0.131692465096 | 0.153746307243 | 1832／1832／1832 | 85／1 |
| 3 | 0.0003 | 0.075033659932 | 0.095960279306 | 1832／1832／1832 | 86／0 |
| 4 | 0.0003 | 0.040568199605 | 0.059717153575 | 1832／1832／1832 | 86／0 |
| 5 | 0.0003 | 0.026260165069 | 0.044142350381 | 1832／1832／1832 | 86／0 |
| 6 | 0.0003 | 0.019903150864 | 0.036415524943 | 1832／1832／1832 | 86／0 |
| 7 | 0.0003 | 0.021023892872 | 0.038009776863 | 1832／1832／1832 | 86／0 |
| 8 | 0.0003 | 0.015582252724 | 0.031598687172 | 1832／1832／1832 | 86／0 |
| 9 | 0.0003 | 0.014166664359 | 0.029468096565 | 1832／1832／1832 | 86／0 |
| 10 | 0.0003 | 0.012090736978 | 0.026928859995 | 1832／1832／1832 | 86／0 |
| 11 | 0.0003 | 0.010105210539 | 0.024710384172 | 1832／1832／1832 | 86／0 |
| 12 | 0.0003 | 0.010225733616 | 0.024468051478 | 1832／1832／1832 | 86／0 |
| 13 | 0.0003 | 0.009040206417 | 0.022791726642 | 1832／1832／1832 | 86／0 |
| 14 | 0.0003 | 0.006510291737 | 0.019120422980 | 1832／1832／1832 | 86／0 |
| 15 | 0.0003 | 0.006457087166 | 0.019120232648 | 1832／1832／1832 | 86／0 |

固定fresh AdamW3e-4／wd1e-4／clip5／SmoothL1 beta.05／default GradScaler／seed20262020，15完整epoch、前14轮无验证，无scheduler／best选择或延长，仅last15。正式82440训练图／1290次CNN与point forward／backward尝试＝1289更新＋1AMPskip；native292／5calls＋CF292／5calls后，正式合计83024图／1300次CNN与point调用。

Pref单列3CNN／8个unique RGB／8次CNN图像呈现／1combined backward／0optimizer。Native identity核实另对相同4张ID再次decode，无额外CNN；CNN unique／presentation计数与decode工作分别记录，不把总decode工作写成8。记录中的cached462初始化、233参数0opt不变、464 strictreload、77BN train+1／eval+0和必要stem／neck／point梯度见[保存见证](initialization_record.json)。

| 成本边界 | 秒 |
|---|---:|
| 正式child process | 510.777282781899 |
| 正式child full module entry | 511.987992271781 |
| Pref full module entry，单列 | 18.654737155885 |
| Supervisor pref stage | 19.630569249392 |
| Supervisor source_control stage | 513.620376408100 |
| Whole supervisor process | 534.320081084967 |
| Individual point／isolated operator compute | unknown／null |

Own entry timer早于torch import，full_process计至保存结果工件、最终completed JSON之前并排除downtime；supervisor wall包含child及工件写入。完整phase time／peak在[成本](costs.json)保留。正式peak allocated／reserved8262538752／11345592320bytes，pref630110208／725614592bytes单列；point与孤立operator计算时间未隔离，保持null，不写0。

Pref于2026-10-25T17:42:11.564332+00:00至17:42:31.193869，source control于17:42:31.319008至17:51:04.938140 UTC，两stage complete／exit0。父420829、pref420832、control420870已退出；17:53:12.777850 UTC保存观察中GPU为空。科学版本45de72cbd7f6c0ac1e507f0e357760e77e2a3933、公开runtime版本e19d33184bc3504e4634752a2f9d07e74f56a9fa分别保留在metadata。

本次公开仅规范化保存scalar CSV／group summaries／history／gates／counts／runtime／初始化及source-isolation／resume metadata。完整source合同、5496训练身份数组、RGB、weights、features、瞬时统计／geometry、privatepaths和rawroot配置不公开，不添加审计脚本或执行模型。

## Limitations

**唯一审计保持FAIL，不能改成clean PASS。** 149 checks中148 PASS、1 FAIL；13617个保存数值对比的最大差为1.4210854715202004e-14。唯一失败native_labels_agree_with_existing_saved_normalized_labels对应：source contract的292行没有normalized_progress或y字段，均有physical label triplet。没有观测到label数值不一致；native、旧directFT5、constant和CF保存的292行ID／group／target精确对齐为PASS，但未重算physical／GT归一化，标签派生全链没有被独立认证。必须按有资格限定的保存数字证据理解，科学门槛判定不覆盖这一认证缺口。

[字段存在性supplement](source_label_field_presence_supplement.json)仅保留既有metadata：normalized_progress_present_rows=0、y_present_rows=0、physical_label_triplet_present_rows=292，physical_label_values_read_or_recomputed=false。唯一数值审计执行1次；supplement仅metadata存在性，不是第二审计，原FAIL和数值记录均保留。

15E对旧directFT5的5E是3倍raw-image／backward-attempt预算，不是matched-budget的颜色算子因果收益。旧baseline只读取保存scalar，不重新训练或forward。一个seed、重复使用的73个源layout×4state与同图固定RGswap，不是新独立域或盲测；channel permutation不代表新物理光照、材质、sensor或表结构，也不证明color cue是既有错误的原因。

CF通过mean／tail／drift门槛仍有大错及>5净新增3，不保证完全color invariance或真实域收益。cached462、strictreload464、BN77及全部233参数零更新保持、formal fresh optimizer／Scaler等仅保存见证与冻结代码；没有fresh检查权重内容。CPU14及GPUempty也为既有保存见证，本次投影与审计不重跑测试或GPU query。point／operator-only未知时间和pref额外identity redecodes按真实计数边界保留。

本控制不是正式V32或泛化候选，不认证标签全链、不自动模型promotion、外评、加seed或延长15E。全部旧模型、G1／anchor／teacher／65轮／配对守恒及其他历史结果保持，旧GeoPRR历史pixels／fullweight链未重新认证。
