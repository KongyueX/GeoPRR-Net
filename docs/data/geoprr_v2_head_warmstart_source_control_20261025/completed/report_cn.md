# 固定100步头warmstart＋FT5源控制：尾部改善，primary不保留

**本次warmstart的source-ID尾部更低，但mean略差于directFT5，未满足预注册primary保留条件：NOT_RETAINED_MEAN_CRITERION_FAILED，retain=false。** 同一292行上，ID mean6.155632265%FS对direct6.132800788%FS，tail21.965393548%FS对25.669996335%FS。原constant0.5双学习门槛已通过，因此同一lastFT5按协议仅评source-OOD220一次；其mean8.930389222／tail25.628879734／max45.643216372%FS只作本次源留出结果。Industrial／RF所有调用0，旧direct baseline没有源OOD结果，不补造比较。唯一独立保存数字／执行核对64／64 PASS、0失败，结论PASS_WITH_QUALIFICATIONS。

| 预注册判断 | 保存值／条件 | 结果 |
|---|---|---|
| 原source学习双门槛 | ID mean≤12.504463703%FS且tail≤23.692208053%FS | 通过 |
| ID mean不高于directFT5 | ID mean6.155632265%FS，高于direct6.132800788%FS | 未通过 |
| ID tail严格低于directFT5 | 21.965393548 < 25.669996335%FS | 通过 |
| 总体primary | 原gate通过，且mean不差，且tail更低 | retain=false |

保存mean变化+0.022831477291806124个百分点、tail变化−3.704602786650259个百分点。不能只报tail改善而改写primary；固定100次LP、mean门槛与cohort保持，不追加次数或换阈值重试。

| 同一source-ID292 | DirectFT5 | Warmstart＋FT5 | Constant0.5 |
|---|---:|---:|---:|
| Mean %FS | 6.132800788 | 6.155632265 | 25.008927406 |
| Acc@1 % | 16.095890411 | 14.383561644 | 2.397260274 |
| Acc@2 % | 28.424657534 | 26.369863014 | 4.794520548 |
| Acc@5 % | 60.958904110 | 54.109589041 | 8.219178082 |
| Median %FS | 3.886425029 | 4.410328716 | 24.896581471 |
| p90 %FS | 12.683204412 | 12.719745040 | 44.168485999 |
| p95 %FS | 17.504554670 | 17.075672448 | 47.623561192 |
| Tail10 %FS | 25.669996335 | 21.965393548 | 47.384416106 |
| Max %FS | 57.870632410 | 48.490351439 | 49.942222197 |
| >1／2／5总数 | 245／209／114 | 250／215／134 | 285／278／268 |
| 输入／scalar失败 | 0／0 | 0／0 | 0／0 |
| Coverage | 1 | 1 | 1 |

| 阈值 %FS | Direct >阈值 | Warmstart >阈值 | 救回 | 新增 | 净新增 |
|---|---:|---:|---:|---:|---:|
| 1 | 245 | 250 | 34 | 39 | +5 |
| 2 | 209 | 215 | 50 | 56 | +6 |
| 5 | 114 | 134 | 54 | 74 | +20 |

救回为direct>阈值、warmstart≤阈值，新增为反向跨越。逐图error改善149／恶化143／相同0，同时>1／2／5净新增5／6／20；这与tail／max降低同时保留。DirectFT5完整基线来自[前一独立控制](../../geoprr_v2_needle_holdout_source_control_20261025/completed/report_cn.md)，同292行身份、group、scalar标签配对；没有重新训练或补评旧baseline。

| 本次同lastFT5源OOD | 保存结果 |
|---|---:|
| 图数 | 220 |
| layout／group数 | 55 |
| Mean %FS | 8.930389222 |
| Acc@1 % | 11.363636364 |
| Acc@2 % | 19.545454545 |
| Acc@5 % | 37.727272727 |
| Median %FS | 7.279267907 |
| p90 %FS | 19.239607826 |
| p95 %FS | 24.001024663 |
| Tail10 %FS | 25.628879734 |
| Max %FS | 45.643216372 |
| >1／2／5总数 | 195／177／137 |
| 输入／scalar失败 | 0／0 |
| Coverage | 1 |

Source-OOD只在原constant gate通过后，对同一lastFT5一次220图／4 CNN调用，既不参与训练也不选择LP次数、epoch或checkpoint。原directFT5的source-OOD未观测，无法给出成对增益；本次不追加其前向。三份完整[ID292](source_id_predictions.csv.gz)、[constant292](constant_source_id_predictions.csv.gz)、[OOD220](source_ood_predictions.csv.gz) scalar CSV、各73／73／55组summary及CSV保留。合法0／1输出保持有效，本次三个人口的边界饱和预测数均0；失败按完整分母error1，不删行。

## 固定学习过程与预算

完全复用已冻结source packet：两seen针型5496训练RGB／687parent（2748 original＋2748 roll），ID292／73layout、metadata字典序留出针型OOD220／55layout，不做新factor扫描、名单／RGB／标签／seed／权重下载。表示不变：完整fresh cached ImageNet MNV4／GAP1／conv_head／norm_head／semantic1280／同一新Linear1280→1／FP32sigmoid，8435793参数／233参数tensor／464state；RGB512／output128／nativeROI／LINEAR／ImageNet／原photo，唯一SmoothL1(beta=.05)。

LP先冻结整个encoder并eval，包括77BN的affine与buffer；训练5496图各提取一次detachedF16语义特征，不shuffle。cache只在server device memory，不落盘或导出。只对同一point做固定100次全5496特征AdamW更新。LP保存100 updates／0AMPskip、462个encoder state不变、point参数改变；100个逐次update前的aggregate loss完整保留在[LP记录](LP_aggregate.json)。update1前loss0.256588339805603，update100前0.2515392005443573；这是固定预算记录，不是额外post-update评估或LP收敛证明。

FT继承这一LP训练过的同头，丢弃LP optimizer／scaler，只恢复fresh初始化后保存的RNG，不重置head；全CNN／point恢复可训、77BN正常source更新，fresh全参数AdamW／defaultGradScaler做原FT5。全CNN反传尝试仍430次，实际426更新＋4正常AMPskip；directFT5为430更新／0skip，两者不描述为相同有效更新数或相同总计算。无中途验证、scheduler、best选择、延长或超参搜索。

| 正式阶段 | CNN图／调用 | Point特征呈现或图／调用 | CNN backward | Point-only backward | 更新／AMPskip |
|---|---|---|---:|---:|---|
| LP cache提取 | 5496／86 | 0／0 | 0 | 0 | 0／0 |
| LP point100 | 0／0 | 549600／100 | 0 | 100 | 100／0 |
| FT5 | 27480／430 | 27480／430 | 430 | 0 | 426／4 |
| Source-ID一次 | 292／5 | 292／5 | 0 | 0 | 0／0 |
| Source-OOD一次 | 220／4 | 220／4 | 0 | 0 | 0／0 |
| 正式合计 | 33488／525 | 577592／539 | 430 | 100 | 分阶段如上 |
| Industrial／RF | 0／0 | 0／0 | 0 | 0 | 0／0 |

总backward530＝100 point-only＋430 fullCNN；point总577592是cached feature呈现与FT／eval图的合计，不是577592张RGB或CNN forward。额外提取5496图和100次head更新不是免费计算。

| FT epoch | LR | Train SmoothL1 | Train NMAE normalized | CNN／Point调用 | 更新／AMPskip |
|---|---:|---:|---:|---|---|
| 1 | 0.0003 | 0.221906560065 | 0.245658833734 | 86／86 | 82／4 |
| 2 | 0.0003 | 0.133309168879 | 0.155474902935 | 86／86 | 86／0 |
| 3 | 0.0003 | 0.081272175712 | 0.102656377889 | 86／86 | 86／0 |
| 4 | 0.0003 | 0.047498441343 | 0.067278957566 | 86／86 | 86／0 |
| 5 | 0.0003 | 0.030933589112 | 0.049358284912 | 86／86 | 86／0 |

Pref另计3CNN／8张不同既有图、3point调用／8example、2 point backward／1CNN backward／0optimizer，source-ID-only、不看OOD。保存见证支持LP encoder grad none／462 state保持／BN+0／point梯度有限非零，FT stem／neck／point梯度有限非零、77BN+1／eval额外0，两back间清梯度、233参数零更新不变、464 strictreload和ID pipeline identity。正式fresh重建不带pref状态；这些是保存见证与code／metadata一致性，不是审计新开checkpoint、cached weights或cache feature内容验证。

## 成本与启动记录

| 保存phase | 秒 | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|
| initialization | 1.698614702 | 68199424 | 79691776 |
| LP_cache | 38.421321146 | 1558400512 | 2279604224 |
| LP_point | 0.362781789 | 99433984 | 2279604224 |
| FT_epoch_1 | 41.648653850 | 8262538240 | 11370758144 |
| FT_epoch_2 | 31.237419415 | 8262538240 | 11370758144 |
| FT_epoch_3 | 29.618203562 | 8262538240 | 11370758144 |
| FT_epoch_4 | 29.840540681 | 8262538240 | 11370758144 |
| FT_epoch_5 | 30.808095921 | 8262538240 | 11370758144 |
| source_id | 5.286944840 | 1805003776 | 11370758144 |
| source_ood | 4.784400802 | 1805003776 | 11370758144 |

| 计时边界 | 秒 |
|---|---:|
| 正式child process | 215.221978381276 |
| 正式child own module-entry full | 216.377160329372 |
| Pref supervisor measured stage wall | 21.367899693549 |
| Control supervisor measured stage wall | 217.468197938055 |
| 正常恢复后supervisor process | 239.490751828998 |
| Pref起点至control结束 | 238.931568 |
| 初始zero-stage failed bootstrap，单列 | 0.277610253543 |
| 独立point-forward计时 | unknown／null |

本module-entry计时已在torch／project imports之前建立，正式child full216.377160329秒计至保存结果工件、最终completed.json之前；supervisor wall覆盖child退出，正常pipeline239.490751829秒。LP_cache38.421321146秒和LP_point完整100更新phase0.362781789秒均包含在正式流程，不把phase时间当独立point forward。各phase point-only计时未单列，保持null而非0。正式peak allocated／reserved8262538240／11370758144bytes；pref663647744／805306368bytes单列。所有真实边界见[成本](costs.json)，不拼入bootstrap等待间隔或混作GPU训练时间。

初始207642cca0f4545ec7181bcbdd2dec0ea148c7b9在冻结snapshot所需metadata部署检查中触发FileNotFoundError：old parent416345、steps为空、尚无reader／pref，model／GPU／weights／RGB调用0。原失败queue／config／protocol／PID及部分snapshot保留于同output的bootstrap_failure。95ed5da343391cfe512a8b5aaed2fbd335316bf5仅修runner和bootstrap检查；root显式--recover-unstarted只接受同合同、zero-stage、旧PID退出／GPU空的边界，主模型／科学config／数据／seed／budget不变。这不是训练resume。新增3项bootstrap CPU检查通过，原10项phase检查未重跑。[规范化启动记录](bootstrap_recovery_record.json)不含raw traceback或私有绝对路径。

正常parent416885、pref416890、control416928串行执行，pref15:44:20.680994→15:44:42.047890 UTC、control15:44:42.145396→15:48:19.612562 UTC（北京时间2026-10-25 23:48:19），两stage complete／exit0。15:49:27.095919 UTC保存完成观察中父子已退出、GPU空闲。初始failed bootstrap实际0.277610254秒单列，启动等待不计入GPU训练。

独立审计[64／64 PASS_WITH_QUALIFICATIONS](numeric_execution_audit.json)保留四项qualification：保存state／静态执行边界、冻结metadata来源、计时和父保存GPU观察、单个有限候选的结论范围。本次公开仅规范化保存scalar／metadata，完整5496训练名单、source arrays、cache features、RGB、weights、geometry、privatepaths、commands与raw traceback保持私有。

## Limitations

本次只说明固定source cohort和LP100＋FT5配方的tail／max更低，ID mean略差且阈值新增错误更多，未满足预注册primary。不据此推断所有LP-FT无效或feature distortion已被证实，也不从相关理论推出本任务收益；100是固定资源上限，不是原论文默认或收敛保证。source-ID已被前一控制使用，不能称新盲测；source-OOD只是三个合成needle的一个固定bank成员，其他layout因素可能混杂，直接baseline没有OOD结果。

额外LP处理与优化属于本控制成本，非相同总计算比较；单seed不支持SD、显著性或整体泛化。Industrial／RF全0，没有真实外域结果。same-head继承、RNG-only恢复、fresh optimizer／scaler及无feature落盘来自冻结代码与保存metadata，不新增checkpoint内容保证。保持mean门槛、100LP、FT5和所有旧资产，不改次数或阈值再试，不自动全量训练或升级模型。
