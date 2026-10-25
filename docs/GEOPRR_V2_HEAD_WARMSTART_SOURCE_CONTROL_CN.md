# 先校准读数头再共同学习：有界源控制

唯一问题：在相同430次完整CNN反传预算下，先用冻结ImageNet表示校准同一个Linear1头，是否改善随后源ID的平均与尾部学习？模型、head、pool、norm、source数据与FT优化不变；只检验同一新run的头初始化流程，不是正式V32、LP-FT论文复现或已证实的泛化方法。

Kumar等ICLR2022的[LP-FT研究](https://arxiv.org/abs/2202.10054)提出先拟合线性头再全量微调；其回归理论简化、实际OOD证据为分类，并依赖较好的预训练特征。这里不认为已证明表征被破坏，也不从该论文推出仪表收益。V15实际冻结的是source训练过的v6/960维GAP，而非freshImageNet1280；V16另起fresh并未继承V15头，不能据二者串行关系认定真正LP→FT已做。V24 source原2.18%并非完全未学，V27非有限失败也不足以判定所有固定统计策略。L2-SP会引入另一个约束系数，本控制不采用它、不加超参搜索。

## 固定源人口、LP与FT

完全复用上一控制服务器已冻结的5496训练RGB／687parent及source-ID292／73layout、源OOD220／55layout：两seen针型训练、metadata字典序第三种留出。无需新factor扫描、source名单、RGB、标签、seed或权重下载，旧14494／原val1506不读，不改变旧split。数据仍nativeROI→LINEAR512／ImageNet，output128仅API，原photo及geometryFalse/pairNone。

fresh既有cached完整MNV4/GAP1/semantic1280/newLinear1280→1 std.001bias0，FP32sigmoid。LP先冻结整个encoder并eval（包括77BN及其affine/buffers），只用训练5496图各一次、同原photo、loader seed20262020、不shuffle提取detachedF16特征。cache仅server device内存，不保存或导出。原point在全5496特征上固定100次AdamW3e-4/wd1e-4/clip5/defaultAMP尝试，SmoothL1beta.05、pointF16/scalarF32；每次forward5496，记录updates/AMPskip，不按loss改次数、不读验证。100是本控制资源上限，未声称LP收敛或论文默认值。

继承同一头，丢弃LPoptimizer/scaler；恢复fresh初始化后记录的随机状态（只恢复RNG、不重置学到的头），整个CNN和point恢复可训、BN恢复正常source更新。fresh全参数AdamW3e-4/wd1e-4/clip5/defaultGradScaler、同seed20262020，5完整FT epoch，train64末56、workers4、每图每E一次、无中途验证／scheduler／best选择／延长。430次全CNN反传与前一控制相同；额外LP计算不当免费，也不把结果归因于已测量的feature distortion。

## 门槛与计算成本

lastFT5只评同292行ID一次。原constant0.5双门槛完全保持：mean和ceil30 tail各≤同人口constant的一半，失败按error1保留分母；未过正常停止、OOD零RGB/forward。通过后同lastFT5源OOD220／4calls一次，真实Industrial/RF全0，不补评旧baseline的源OOD。只有原门槛通过、IDmean不高于directFT5的6.13280078773%且tail低于25.66999633486%，才保留此有限warmstart候选；失败不扫LP次数、学习率、seed、cohort或阈值，不推翻全部LP-FT理论。

正式LPcache为5496CNN图／86calls／0CNNback；LPpoint为100calls／549600次cache特征呈现／100point-onlyback。FT为27480图／430CNN和pointcalls／430CNNback。加ID后CNN33268图／521calls，point577372呈现／535calls；若过门槛加OOD，则CNN33488／525，point577592／539。总backward530=100point-only＋430fullCNN；各阶段updates＋AMPskip分别100／430。源uniqueRGB仍5496，cached feature呈现不是新图或CNNforward。

## 新检查及执行边界

只新增CPU阶段冻结/恢复、梯度模式、RNG/optimizer边界、不同CNN/point计数检查；复用旧sourcepacket和gate，不重复旧selector、GT、数学、模型构造或tests。一次新source-ID-only零更新pref恰3CNN／8张不同既有图／2pointback／1CNNback／0optimizer：LP2seen original首parent slot0、FT4seen original+roll同parent slot3、ID2seen各一；pref不看OOD。LP应无encodergrad／BNcounter更新且462encoderstate保持，FT必要stem/neck/point梯度有限非零且77BN+1、eval额外0；两back间清梯度，233参数0opt不变、464strictreload、cached462初值保持、真实512/F16semanticpoint/F32scalar/IDpipeline identity。pref固定loss128/grad128，正式fresh重建，不带pref状态。

代码/协议/config/NEWCPU证据先existingmain冻结push、实际旧父子退出且GPU空后才唯一pref→LP100→FT5→IDgate→条件源OOD；不改live、不额外GPU或两个真实外域。自身module-entry计时器必须早于torch import，supervisor wall与实际phase time/peak分开，未知point-only时间不填0。完整source合同/训练arrays/特征值/RGB/weights/geometry/privatepaths留server private，公开仅规范化保存标量及metadata。失败保留partial不fresh覆盖或自动延长，完成后唯一新鲜独立CPU保存数字核对归档。

## 完成记录

LP100＋FT5完整结束，原constant0.5半mean／半tail学习gate通过，但预注册primary为NOT_RETAINED_MEAN_CRITERION_FAILED、retain=false。ID mean6.155632265021%FS略高于directFT5的6.132800787730%FS，tail21.965393548210%FS低于25.669996334861%FS；>1／2／5救回34／50／54、新增39／56／74，净新增5／6／20。保留tail收益、mean代价和阈值退化，不只报tail，不调整mean门槛或LP100次数再试。

同一lastFT5在原gate通过后只评源OOD220一次，mean8.930389222435／tail25.628879734061／max45.643216371536%FS。旧directbaseline没有源OOD结果，不补造或补评；Industrial／RF所有images／calls／backward／updates0。LP100更新／0skip、462encoderstate保持且head改变；FT430 fullCNN backward尝试＝426更新＋4正常AMPskip。正式CNN33488图／525calls，point577592 cached feature／raw图呈现、539calls，backward530＝100 point-only＋430CNN；pref另3CNN8不同图、2pointback1CNNback0opt。cache只server device memory，不落盘／导出；同头、RNG-only恢复、fresh optimizer／scaler与state见证只由保存code／metadata支持，不增加fresh checkpoint内容核实。

初始207642cca0f4545ec7181bcbdd2dec0ea148c7b9在snapshot metadata部署阶段FileNotFound，old parent416345／steps空，0model／GPU／weights／RGB，原queue／config／protocol／PID及partial snapshot保留。实际修复95ed5da343391cfe512a8b5aaed2fbd335316bf5只改runner／bootstrap tests；root显式--recover-unstarted在同合同、无reader／pref／模型stage、旧PID退出／GPU空时恢复未开始的启动，主模型／科学配置／数据／seed／budget不变，不是训练resume。额外3项bootstrap tests通过，原10项phase tests未重跑。

正常parent416885／pref416890／control416928，两stage complete／exit0；control结束2026-10-25T15:48:19.612562+00:00（北京时间23:48:19），15:49:27.095919 UTC保存观察中父子gone／GPU空。自身module-entry计时早于torch／project imports，正式child full216.377160329秒计至保存工件、最终completed.json之前；正常supervisor pipeline239.490751829秒，pref起点至control结束238.931568秒。LP cache38.421321146秒／LP100完整优化phase0.362781789秒均为额外计算，point独立计时null，phase／pipeline／peak分别保留；failed bootstrap0.277610254秒单列，等待间隔不算GPU训练。

唯一独立保存数字／执行核对64／64 PASS、0失败，总PASS_WITH_QUALIFICATIONS，四项qualification在[审计](data/geoprr_v2_head_warmstart_source_control_20261025/completed/numeric_execution_audit.json)完整保留。[报告](data/geoprr_v2_head_warmstart_source_control_20261025/completed/report_cn.md)、[指标](data/geoprr_v2_head_warmstart_source_control_20261025/completed/metrics.json)、[primary](data/geoprr_v2_head_warmstart_source_control_20261025/completed/primary_comparison.json)、三份完整scalar CSV（ID292／constant292／OOD220）及各组summary／CSV、LP100aggregate loss／FT5 history规范化归档。根7入口＋[completed26项](data/geoprr_v2_head_warmstart_source_control_20261025/completed/public_allowlist.json)，[总allowlist](data/geoprr_v2_head_warmstart_source_control_20261025/public_allowlist.json)33项。本次公开只保存规范化scalar／metadata，source5496 arrays、cachefeatures、RGB／weights／geometry／privatepaths／raw traceback留私有；全部旧资产保留，不自动扩大训练或升级模型。

## Limitations

预训练特征适合本任务的程度仍未知；额外LP数据处理及优化是本控制的一部分，不是同总成本比较。source-ID已用于上一控制与诊断，不能称新盲测或显著性；source-OOD只是三个合成needle中的一个固定bank成员，其他layout因素混杂。源ID改善或单域结果不等于整体泛化成功，不自动升级模型、拼checkpoint或开长训练。


启动metadata先于所有output写入检查；默认不恢复已有输出。若只在snapshot部署阶段失败且全部模型stage均未开始，root可显式恢复该zero-stage bootstrap：同protocol/config/source合同、steps空、无reader/pref、旧PID退出且GPU空，先保留原queue/protocol/config/PID/部分snapshot于同output/bootstrap_failure，再继续尚未开始的启动。任何训练或pref partial均不适用，不覆盖模型、改budget或自动resume。
