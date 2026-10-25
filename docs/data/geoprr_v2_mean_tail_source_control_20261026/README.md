# Mean＋batch-tail source风险控制

**科学结果NOT_RETAINED：原constant双学习gate通过，但新的native联合验收五项全false。** Native mean3.855632551／tail14.186078385／max58.501848578%FS，高于identity参考2.543121824／11.800955331／53.797036409%；>5错误55对16。参考source-ID error≤1%FS的102行中新增15个>5错误、涉及14groups；这仅是本292行source-ID准确参考集，不是旧ordinary1495／1497。

[完整报告](completed/report_cn.md)、[指标](completed/metrics.json)、[native primary](completed/native_primary.json)与[最终唯一审计](completed/numeric_execution_audit.json)保留171／171 PASS、0FAIL、12050numeric／maxdiff1.4210854715202004e-14（abs1e-12）及资格。Identity→candidate图像91改善／201恶化、groupmean11／62、groupmax16／57；>1／2／5新增80／110／45、救回29／30／6，>1／2／5总数241／190／55对参考190／110／16。

唯一loss聚合为`.5 × mean(per-image SmoothL1β.05) + .5 × mean(top ceil(.1B) per-image loss)`。B64取7、last56取6、pref4取1，每E601 upweighted slots，全部5496图仍贡献mean项。[Objective evidence](completed/objective_saved_evidence.json)及metadata明确：原history键source_training_smooth_l1保存的是composite batch risk，不是primitive mean或whole-population tail；unweighted trainNMAE不变。标准top-k风险聚合不声称新理论或ATk／DORO论文复现，未知noise不被描述为已发现。

固定原生RGB、模型、source packet／photo／seed／15E，actual每E RGB5496／GBR0／BRG0，nominal slots[1832,1832,1832]不是三颜色训练。Scheduled1290 CNN-point backward尝试与identity相同，candidate实际1287updates＋3正常AMPskip，对identity1289＋1；不补更新、不延长。正式82440训练图＋native292＝82732图／1295 CNN及point calls，pref另8unique／3CNN1back0opt，ID4图repeat decode计CPUwall。

[Candidate CF状态](completed/candidate_counterfactual_status.json)严格为null／0calls、无candidateCF文件；reference_identity_source_counterfactual_predictions.csv.gz是已存identity参考，不能当本候选CF成绩。不补CF或旧forward／weights，source03／Industrial／RF全部0RGB／model。完整candidate native／constant292、reference native／CF292 scalar CSV、native292 pairdata、scalar groups与summary保留；完整阈值／cohort ID名单省略，不删逐图CSV。

[实际成本](completed/costs.json)保留train455.614306495秒、nativepipeline5.047001932秒，formalentry467.752068713秒，pref／source supervisor18.719613757／469.070528772秒、whole488.628538150秒，peak8262538752／11345592320bytes。Point／isolated loss／operator单独成本unknown／null，前后context／timer／defaultScaler／cached／BN／strict等只saved witness＋code，不新开weights内容。

科学743acb16b486f1f494c3b1f49fa7c4a41848887d、runtime8e0bd5f806bd279751025b04b6ccc6e87520c7fd已冻结。Provided-label／rawlabel派生资格见报告Limitations；旧C3 149／148／1及旧目录保持，不新增GT审核。根目录7项＋独立协议1项＋[completed31项](completed/public_allowlist.json)，[总显式清单](public_allowlist.json)39项。本次只规范化保存scalar／metadata，完整合同／population／trainarrays／RGB／weights／features／geometry／privatepaths私有；不改gate、不加预算、loss扫描或promotion。
