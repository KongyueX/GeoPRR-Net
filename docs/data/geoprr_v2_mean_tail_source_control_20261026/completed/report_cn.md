# Mean＋batch-tail source控制：原学习gate通过，联合验收五项均未通过

**科学判定NOT_RETAINED。** Candidate native mean3.855632551／tail14.186078385／max58.501848578%FS，均高于identity参考的2.543121824／11.800955331／53.797036409%FS；>5错误55对16，准确参考集新增大错guard也失败。原constant双学习gate通过，但新的native联合验收五项全false，按协议停止，candidate CF为null／0calls、没有候选CF结果文件。最终唯一provided-label保存数字审计171／171 PASS、0FAIL，12050数值对比maxdiff1.4210854715202004e-14（abs tolerance1e-12）。

| Native联合条件 | Identity参考／固定上限 | Mean-tail candidate | 判定 |
|---|---:|---:|---|
| Mean不高于参考 %FS | 2.543121824 | 3.855632551 | 未通过 |
| ceil30 tail严格低于参考 %FS | 11.800955331 | 14.186078385 | 未通过 |
| Maximum不高于参考 %FS | 53.797036409 | 58.501848578 | 未通过 |
| >5%FS错误计数不增加 | 16 | 55 | 未通过 |
| 参考accurate集无新增>5错误 | 0 | 15 | 未通过 |

本292行source-ID中，identity参考error≤1%FS的102行内出现15个新的>5%FS错误，涉及14个group。这是本source-ID的准确参考子集，不是旧ordinary1495／1497人口；完整292行scalar与pair CSV保留，资格和完整身份数组范围见Limitations。

| 同一native292行 | Identity参考 | Mean-tail candidate | Constant0.5 |
|---|---:|---:|---:|
| Mean %FS | 2.543121824 | 3.855632551 | 25.008927406 |
| Acc@1 % | 34.931506849 | 17.465753425 | 2.397260274 |
| Acc@2 % | 62.328767123 | 34.931506849 | 4.794520548 |
| Acc@5 % | 94.520547945 | 81.164383562 | 8.219178082 |
| Median %FS | 1.549451053 | 2.838931605 | 24.896581471 |
| p90 %FS | 4.091511518 | 6.806516163 | 44.168485999 |
| p95 %FS | 5.404334664 | 8.725114092 | 47.623561192 |
| ceil30 Tail10 %FS | 11.800955331 | 14.186078385 | 47.384416106 |
| Max %FS | 53.797036409 | 58.501848578 | 49.942222197 |
| >1总数 | 190 | 241 | 285 |
| >2总数 | 110 | 190 | 278 |
| >5总数 | 16 | 55 | 268 |
| Coverage | 1 | 1 | 1 |
| 输入／scalar失败 | 0／0 | 0／0 | 0／0 |

所有native人口均完整292行／73layout×4state。原constant门槛mean≤12.504463703%FS且tail≤23.692208053%FS已经满足，不能以这一最低学习条件替代新的联合验收。Candidate max ID示例geoprr_roll_v22_validation_g0061_s03，valid native精确0／1预测计数为0；合法饱和值仍按现有规则有效，失败保留error1，不删行。

| Identity→mean-tail native配对 | 改善 | 恶化 | 相同 |
|---|---:|---:|---:|
| 图像absolute error | 91 | 201 | 0 |
| Group mean error | 11 | 62 | 0 |
| Group maximum error | 16 | 57 | 0 |

| 阈值 %FS | 参考>阈值 | Candidate>阈值 | 新增 | 救回 |
|---|---:|---:|---:|---:|
| 1 | 190 | 241 | 80 | 29 |
| 2 | 110 | 190 | 110 | 30 |
| 5 | 16 | 55 | 45 | 6 |

图像91改善／201恶化，group mean11／62、group max16／57；>1／2／5新增80／110／45、救回29／30／6。新增为参考≤阈值而candidate>阈值，救回为反向跨越；保存的逐图pair及group比较完整保留。1／2计数只作诊断，未新增为验收旋钮，也不以一个重新定义的source risk放宽失败gate。

## 唯一objective及history语义

每图仍是原SmoothL1(beta=.05)，唯一变化为固定batch聚合：

`L = 0.5 × mean(loss_i) + 0.5 × mean(largest ceil(0.1 × B) per-image loss_i)`

实际B64取k7、末B56取k6、prefB4取k1；每E601个upweighted slots。全部5496图都通过mean项贡献，不删除、重采样或遗漏非top-k图。权重.5和fraction.1固定，无lambda／beta／risk扫描。Batch-local SmoothL1 tail与完整292人口的absolute-error tail分别定义。

原history键source_training_smooth_l1保留，但数值是按图数加权汇总的batch composite risk，不是primitive mean SmoothL1，也不是整个人口tail；source_training_nmae仍是原unweighted absolute error。完整15E raw history及[objective evidence](objective_saved_evidence.json)保留，训练损失不得按旧primitive mean loss数值排名。

| Epoch | LR | Composite batch-risk，原history键 | Unweighted train NMAE normalized | Nominal slots | 实际RGB／GBR／BRG | 更新／AMPskip |
|---|---:|---:|---:|---|---|---|
| 1 | 0.0003 | 0.365210913675 | 0.279253289169 | 1832／1832／1832 | 5496／0／0 | 85／1 |
| 2 | 0.0003 | 0.350298456533 | 0.251045208831 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 3 | 0.0003 | 0.307400679528 | 0.185227510537 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 4 | 0.0003 | 0.260369840668 | 0.143271287890 | 1832／1832／1832 | 5496／0／0 | 85／1 |
| 5 | 0.0003 | 0.195074338949 | 0.104658492094 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 6 | 0.0003 | 0.117425265249 | 0.066769976847 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 7 | 0.0003 | 0.094009766195 | 0.055900213717 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 8 | 0.0003 | 0.071150035930 | 0.046756660010 | 1832／1832／1832 | 5496／0／0 | 85／1 |
| 9 | 0.0003 | 0.059523936659 | 0.041249103248 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 10 | 0.0003 | 0.054843038494 | 0.038406367841 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 11 | 0.0003 | 0.048676022140 | 0.035289276136 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 12 | 0.0003 | 0.045736884332 | 0.033798597699 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 13 | 0.0003 | 0.045665350912 | 0.033219970101 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 14 | 0.0003 | 0.037624649855 | 0.029505449568 | 1832／1832／1832 | 5496／0／0 | 86／0 |
| 15 | 0.0003 | 0.033821302112 | 0.029021960245 | 1832／1832／1832 | 5496／0／0 | 86／0 |

15E固定82440训练图／1290 CNN与point backward尝试，实际1287更新＋3正常AMPskip；identity参考为1289＋1。Scheduled budget相同，实际有效更新数不同，不补skip、不延长、无scheduler／中途验证／best选择，仅last15。每E全部实际RGB5496／GBR0／BRG0，[1832,1832,1832]仅nominal slot bookkeeping，不能解释为C3颜色训练。

模型与输入保持fresh cached MNV4／GAP1／conv_head／norm_head／semantic1280／Linear1 std.001 bias0／FP32sigmoid，8435793参数／233参数tensor／464state、77BN源训练更新／eval冻结；source packet5496图／687parent、ID292／73layout、nativeROI／LINEAR512／ImageNet／原photo、seed20262020、AdamW3e-4／wd1e-4／clip5／defaultfreshScaler及batch64／workers4固定。没有新head／pool／norm、teacher、geometry输入、mask或新label／RGB／weights／source population。

| 人口或文件角色 | 模型调用／结果 |
|---|---|
| Candidate native source-ID292 | 一次292图／5calls，完整native结果 |
| Candidate counterfactual | native joint未过；0RGB／0CNN／0point，metrics=null，无candidate CF文件 |
| Reference identity native／CF | 已存参考scalar，仅provided-label配对，无old forward／weights打开 |
| Source03／Industrial／RF | 0RGB／model／backward／updates，无结果 |

[candidate CF状态](candidate_counterfactual_status.json)明确跳过。文件reference_identity_source_counterfactual_predictions.csv.gz仅为旧identity保存参考，不能当作candidate CF成绩；候选不存在source_counterfactual_predictions.csv.gz或candidate CF summary。当前颜色drift门槛仅诊断，本次因native失败连CF也未执行，不把参考CF填进空值。

正式合计82732图／1295 CNN及point calls、1290back尝试；pref另3CNN／8unique RGB／8 presentations／1combinedback／0optimizer。Pref的同4ID另行decode做native identity，CPU工作计入wall，不声称总decode8。Pref实际composite objective0.424453467130661、B4／k1和必要finite非零stem／neck／point梯度保留为保存证据，正式fresh不carry pref状态。

| 保存成本边界 | 秒 |
|---|---:|
| Formal train保存和 | 455.614306494594 |
| Native pipeline保存和 | 5.047001931816 |
| Formal child process | 465.964059531689 |
| Formal own full entry | 467.752068713307 |
| Pref full entry | 17.890132773668 |
| Supervisor pref stage | 18.719613756984 |
| Supervisor source stage | 469.070528771728 |
| Whole supervisor | 488.628538150340 |
| Point／isolated loss／operator独立成本 | unknown／null |

Formal peak allocated／reserved8262538752／11345592320bytes；pref630110208／725614592bytes单列，完整phase time／peak保留在[成本](costs.json)。Pref2026-10-25T19:53:34.879404→19:53:53.597976 UTC，control19:53:53.727164→20:01:42.796408 UTC，两stage complete／exit0；427975／427978／428016退出，20:15:16保存观察GPU空。科学743acb16b486f1f494c3b1f49fa7c4a41848887d、runtime8e0bd5f806bd279751025b04b6ccc6e87520c7fd分别保留。

本次公开仅规范化完整scalar CSV／groups／summary／history／gates／pairdata及config-run-completed-queue-screen-count-time-peak-init-ref-isolation-resume／objective metadata。Raw threshold／准确参考完整ID列表被省略，完整292行CSV与pairdata不删；fullcontracts、populations、5496训练arrays、RGB、weights、features、geometry与private absolute paths保持私有。

## Limitations

最终唯一provided-label审计171／171 PASS、0FAIL，12050 numeric comparisons最大差1.4210854715202004e-14，按既有abs1e-12容差接受浮点数值；整数计数、ID／group／target、metadata／config／schema按其保存规则精确核对。本次未重算physical label或重做旧identity／C3审计，raw-label派生未新增认证；旧C3仍149／148／1、无新label冲突，旧档案不变。

这是一seed、73source layout×4state及先前已查看的development view，不是独立域、新blindtest、SD、显著性、理论或真实Industrial／RF泛化。102行accurate参考集只来自本292行provided-label source-ID，不能替代旧ordinary1495／1497。当前五项joint失败，不放宽门槛、追加15E／CF、换risk权重／seed或promotion。

Factory／cache／seed／冻结代码和pref支持初始化parity，但未新开旧weights。缓存462／strict464／233参数0opt保持、BN77源+1／eval+0、context隔离、defaultScaler、梯度／dtype与pipeline identity等只保存见证及code／metadata，不是新checkpoint内容或源像素核实。Entry timer早于torch import并包括wrapper reference／metadata工作，full child排除最终completion JSON和downtime；supervisor覆盖startup／child退出，point／孤立loss聚合／operator未单测为null，不能称免费或用wall差倒推净成本。

Batch top-k可能强化异常或不可约高loss，但实际label／input noise未知。本控制不采用DORO去样本／裁剪，不复现ATk／DORO论文或声称新理论；也未证明batch-tail与population-tail定义差异就是失败根因。训练既有augment=True异常路径raise，不把坏图占位继续当作已证实的放大机制，未为此新扫RGB或GT。过去逐图训练loss不能从aggregate history重建，本次未宣称重建。

原constant最小学习gate通过不能抵消五项joint全false；科学结果保持NOT_RETAINED。全部旧正负结果、source合同与checkpoint资产保留，不自动长训、source03或真实外域评估，不新增noise／GT审核。
