# 固定头warmstart源控制

**Source-ID尾部改善，但mean略差于directFT5，预注册primary不保留：NOT_RETAINED_MEAN_CRITERION_FAILED，retain=false。** ID mean6.155632265%FS对direct6.132800788%FS，tail21.965393548%FS对25.669996335%FS；>1／2／5救回34／50／54、新增39／56／74，净新增5／6／20。原constant0.5半mean／半tail学习gate通过，不等于primary通过，不改mean门槛或100LP次数重试。

原gate通过后，同一lastFT5仅评source-OOD220一次，mean8.930389222／tail25.628879734／max45.643216372%FS。旧directFT5无source-OOD结果，不补造比较；Industrial／RF images／CNN／point／backward／updates均0。[完整报告](completed/report_cn.md)、[指标](completed/metrics.json)、[primary](completed/primary_comparison.json)与[64／64独立核对](completed/numeric_execution_audit.json)保留混合结果及PASS_WITH_QUALIFICATIONS；这不是正式V32、整体泛化成功或所有LP-FT的结论。

[冻结协议](../../GEOPRR_V2_HEAD_WARMSTART_SOURCE_CONTROL_CN.md)保持完整MNV4／GAP1280／同一Linear1，复用source packet5496训练RGB、ID292、OOD220。LP encoder冻结并eval，只提取5496图一次detachedF16 cache，随后同头100更新／0skip；462encoderstate保持，head改变。cache只在server device memory，不落盘／导出。继承该head、只恢复保存的RNG，丢弃LP optimizer／scaler，再fresh全参数FT5：430次CNN backward尝试＝426更新＋4正常AMPskip。额外LP不是免费计算，也不是相同有效更新数比较。

正式CNN33488图／525调用，point577592次cache特征或图呈现／539调用；总backward530＝100 point-only＋430全CNN，point呈现不冒称RGB或CNN forward。pref另3CNN／8张不同图、2pointback／1CNNback／0optimizer。cached462、LP state保持、同头／RNG继承与464 strictreload仅为保存见证及code／metadata一致性，未fresh打开checkpoint、weights或feature值。完整三份scalar CSV.gz（292／292／220）、73／73／55组summary／CSV、LP100 aggregate loss和FT5 history均保留。

[成本](completed/costs.json)保留LP cache38.421321146秒、LP100完整优化phase0.362781789秒、FT163.152913429秒和各phase peak；独立point forward时间unknown／null。自身module-entry计时早于torch／project imports，正式child full216.377160329秒至结果工件保存、最终completed.json前；正常supervisor pipeline239.490751829秒，pref起点至control结束238.931568秒。正式peak allocated／reserved8262538240／11370758144bytes；额外LP不当免费。

初始207642cca0f4545ec7181bcbdd2dec0ea148c7b9在snapshot metadata部署检查触发FileNotFoundError，parent416345／steps空、0 model／GPU／weights／RGB。原queue／config／protocol／PID和partial snapshot保留；95ed5da343391cfe512a8b5aaed2fbd335316bf5只改runner及bootstrap检查。root显式--recover-unstarted在同科学合同、旧PID退出／GPU空且未开始任何模型stage时继续启动，主模型／科学config／数据／seed／budget不变，**不是训练resume**。额外3项bootstrap tests通过，原10项phase tests未重跑；[启动记录](completed/bootstrap_recovery_record.json)只含规范化metadata。failed bootstrap0.277610254秒单列，不把等待间隔算GPU训练。

正常parent416885／pref416890／control416928两stage complete／exit0，control结束2026-10-25T15:48:19.612562+00:00（北京时间2026-10-25 23:48:19）；15:49:27.095919 UTC保存观察时父子退出／GPU空闲。根7入口＋[completed26项](completed/public_allowlist.json)，[总显式清单](public_allowlist.json)33项。本次公开仅规范化保存scalar／metadata，source arrays／5496名单／cachefeatures／RGB／weights／geometry／privatepaths／raw traceback不公开；全部旧资产保留，不追加100LP、不改primary门槛、不自动全量训练或升级模型。
