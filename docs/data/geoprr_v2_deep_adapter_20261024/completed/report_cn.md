# V24固定完整主干的深层adapter：完整结果

**RF100平均与严重尾部相对v16改善，但Industrial显著退化，未达到两个外域整体超过v16及旧GeoPRR的目标。** 完整源98轮结束，仅源R选出的best78各冻结外评一次。RF尾部与max也低于旧历史，但mean仍高于旧GeoPRR；>5%FS新增多于救回，不能以尾部收益代替整体精度。全部模型、负结果和固定人口保留，不自动升级默认、不延长当前配方。

| 数据集／模型 | Mean %FS | Acc@1 % | Acc@2 % | Acc@5 % | Tail10 %FS | Max %FS |
|---|---:|---:|---:|---:|---:|---:|
| industrial／v16 | 11.686054 | 13.046595 | 22.293907 | 44.874552 | 45.734073 | 86.481965 |
| industrial／旧GeoPRR历史 | 11.643618 | 9.032258 | 17.491039 | 45.949821 | 43.950870 | 75.324667 |
| industrial／V24 best78 | 23.980845 | 2.365591 | 5.806452 | 16.989247 | 65.664809 | 95.865870 |
| rf100／v16 | 10.164776 | 23.841060 | 37.086093 | 61.589404 | 44.210096 | 54.413070 |
| rf100／旧GeoPRR历史 | 6.607079 | 27.814570 | 45.695364 | 69.536424 | 32.748645 | 60.504565 |
| rf100／V24 best78 | 7.726820 | 9.271523 | 17.218543 | 43.046358 | 25.325275 | 50.892561 |

| 数据集 | N／组 | Median %FS | p90 %FS | p95 %FS | >1／2／5%FS总数 |
|---|---|---:|---:|---:|---|
| industrial | 1395／52 | 18.403316 | 54.873681 | 62.700377 | 1362／1314／1158 |
| rf100 | 151／35 | 6.058630 | 16.284921 | 22.082233 | 137／125／86 |

| 数据集／参照 | 逐图改善／恶化 | 组改善／恶化 | >1救回／新增 | >2救回／新增 | >5救回／新增 |
|---|---|---|---|---|---|
| industrial／v16 | 329／1066 | 8／44 | 30／179 | 58／288 | 64／453 |
| industrial／historical_old_geoprr_clean_source_only | 323／1072 | 21／31 | 24／117 | 42／205 | 68／472 |
| rf100／v16 | 64／87 | 14／21 | 9／31 | 11／41 | 17／45 |
| rf100／historical_old_geoprr_clean_source_only | 48／103 | 11／24 | 6／34 | 7／50 | 12／52 |

救回为参照>阈值、本模型≤阈值；新增为反向跨越。完整同人口保存预测对齐，无输入或scalar失败，coverage=1。RF100 mean/tail相对v16分别降低2.437956／18.884820个百分点，但>5新增45／救回17；对旧GeoPRR mean升高1.119740个百分点、tail降低7.423370个百分点，>5新增52／救回12。Industrial对v16 mean/tail升高12.294791／19.930736个百分点，>5新增453／救回64；对旧历史新增472／救回68。参照v18／v21／v22／v23／v12的完整比较保留在[比较表](comparison.csv)和[指标](metrics.json)，不合成不同checkpoint的最佳域成绩。

| 源checkpoint／人口 | Mean %FS | Tail10 %FS | Max %FS | >1／2／5 |
|---|---:|---:|---:|---|
| best_source／original | 2.180123 | 9.578513 | 54.945527 | 860／486／134 |
| best_source／roll | 3.952898 | 14.119596 | 40.320966 | 410／317／126 |
| last_source／original | 2.163619 | 9.608645 | 53.451502 | 848／464／131 |
| last_source／roll | 4.052144 | 14.748689 | 43.822363 | 389／319／130 |

源best78 R=7.457783%FS，last98 R=7.643274%FS，R仍为原mean＋rollmean＋原tail10＋rolltail10四项normalized error平均，tail严格ceil10%（151／52）。98=78+20，满足min60/max160与完整20轮无严格改善的停止规则，LR在74／87／96后下降未重置counter，无resume。原源及roll源精度均明显弱于直接完整语义源参照，不能用本轮RF局部收益称源拟合已解决。

| 源checkpoint | ordinary1495 mean %FS | ordinary1497 mean %FS | severe6 mean %FS |
|---|---:|---:|---:|
| best_source | 2.111608 | 2.151475 | 15.585863 |
| last_source | 2.097040 | 2.132916 | 16.029689 |

| 固定源sample | best78 error %FS | last98 error %FS |
|---|---:|---:|
| sync_15913 | 23.986387 | 26.918995 |
| sync_2810 | 1.217639 | 1.394123 |
| sync_3172 | 24.609131 | 18.549943 |
| sync_338 | 5.263513 | 6.332070 |
| sync_3663 | 15.331675 | 16.343386 |
| sync_4025 | 12.520242 | 12.421340 |
| sync_4658 | 1.169280 | 0.270424 |
| sync_615 | 5.694823 | 5.677802 |
| sync_8195 | 54.205006 | 53.451502 |
| sync_9 | 11.877690 | 11.315801 |
| sync_9323 | 1.053190 | 0.924104 |

四份完整source best／last标量CSV、各组与zero／nonzero／nearzero人口均保留。RF100固定6／41／46／140的V24误差分别为9.421447／3.369932／12.902997／8.771291%FS；这些是保存结果回顾，未用于本轮标签、阈值或配置选择。Industrial与RF的精确zero／nonzero／nearzero、所有focus连续误差与旧模型参照均见metrics。原14494／1506／4000、ordinary1495／1497／severe6与重点名单没有改变。

V24实际8910793参数／241个参数tensor／472 state；原8434512参数／462 state、77BN全部98轮直接保持相等，原参数grad为0、encoder始终eval。只有两个深层pointwise GELU残差adapter与新point共476281参数／10tensor训练；固定后续CNN仍保留输入autograd，第一个adapter的up梯度在实际源见证中有限非零。两个adapter在每次CNN均实际调用，不能把原paramgrad0写成没有反传或零成本。完整semantic1280保留，无旧task checkpoint／query／FPN／router／teacher／GT角色／输出融合。

源数据完全复用V22的22686unique／34974每轮呈现、原1506＋roll512，256LINEAR/ImageNet与旧光度和scalar SmoothL1(beta=.05)保持。新模型的全部署合同不同，只复用RGB／源人口子合同。roll512独立layout／ID／group／声明seed但sharedHDR，原20096 metadata缺seed，旧1506scene guard保持；不认证完整历史seed链或新域盲测。

source完成2026-10-24T12:40:27.573366UTC，Industrial于12:40:27.661022启动、12:40:39.392568结束；RF于12:40:39.460694启动、12:40:50.095903结束。各一次冻结同一source-best78，last只源；目标训练、总体统计拟合、BN适配、checkpoint／epoch／早停／训练超参／阈值／ROI／温度选择均0，RFscalar由四点离线派生，输入／scalar失败按完整原图分母error1。

实际训练3427452图／53606次CNN与loss backward尝试＝53586有效更新＋20AMPskip；validation197764图／3136次CNN，外域1546图／25次CNN／0back／0update，总3626762图／56767次CNN。每个adapter的calls／images与对应CNN完全相同。新pref单列3CNN／8图／1combinedback／0opt；正式fresh重建，未沿用prefBN。原462初始state从完整cachedImageNet保留、完整model／optimizer／scheduler／scaler／RNG／history保存。审计未加载checkpoint内容，保存机制由执行代码与保存metadata支持。

训练／验证pipeline8439.410359／422.064307秒，source process／full8926.789752／8927.727020秒；source至最后外评supervisor8951.114030秒，含pref8966.034172秒。source peak allocated／reserved664222720／815792128 bytes，旧源生成只引用已有记录，不重复计新渲染；见[成本](costs.json)。

四阶段complete／exit0、父进程退出、收集时GPU空闲。唯一独立CPU保存数字／执行核对72401项通过、0失败、最大差1.4210854715202004e-14；未运行模型、weights、RGB、GT、旧测试或旧审计，见[numeric audit](numeric_execution_audit.json)。[98轮history](history.jsonl)、[best](best_validation.json)、[last](latest_validation.json)及[显式allowlist](public_allowlist.json)保留实际执行证据。

## Limitations

本轮固定完整原表示与BN、两处内部adapter和训练范围共同改变；源欠拟合和域间差异不证明所有adapters或固定表示无效。RF尾部改善不能掩盖主体准确率下降、Industrial退化或相对旧历史mean的差距。两域为已查看开发人口，单seed不支持SD、显著性或普适性；旧GeoPRR历史pixel／完整权重链未重新认证，不能归因SARN／polar／router或编造校准收益。原模型、失败、G1／anchor／65轮等全部保留。rawrun／fullcontracts／完整源数组／RGB／weights／H/C/XYZ/U/M／bbox／features／privatepaths保持私有，只规范化标量工件公开。真人已授权持续独立源域结构研究，单轮结束不扩展当前配方或重复外评。
