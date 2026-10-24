# V25有序读数分布与CDF log-score：完整结果

**V25改善源域精度，但两个外域整体超越的目标未达成，不升级默认。** 完整源60轮结束，仅源R选出的best37各冻结外评一次。Industrial mean21.972040／tail60.546531%FS，明显弱于v16及旧GeoPRR；较V24改善。RF100 mean9.626736／tail42.779526%FS，较v16小幅改善，但低阈值准确率下降、max增大，仍弱于旧GeoPRR；其mean／tail也弱于V24已取得的RF收益。全部结果和旧资产保留。

| 数据集／模型 | Mean %FS | Acc@1 % | Acc@2 % | Acc@5 % | Tail10 %FS | Max %FS |
|---|---:|---:|---:|---:|---:|---:|
| industrial／v16 | 11.686054 | 13.046595 | 22.293907 | 44.874552 | 45.734073 | 86.481965 |
| industrial／旧GeoPRR历史 | 11.643618 | 9.032258 | 17.491039 | 45.949821 | 43.950870 | 75.324667 |
| industrial／V24 best78 | 23.980845 | 2.365591 | 5.806452 | 16.989247 | 65.664809 | 95.865870 |
| industrial／V25 best37 | 21.972040 | 7.885305 | 15.483871 | 34.265233 | 60.546531 | 82.658517 |
| rf100／v16 | 10.164776 | 23.841060 | 37.086093 | 61.589404 | 44.210096 | 54.413070 |
| rf100／旧GeoPRR历史 | 6.607079 | 27.814570 | 45.695364 | 69.536424 | 32.748645 | 60.504565 |
| rf100／V24 best78 | 7.726820 | 9.271523 | 17.218543 | 43.046358 | 25.325275 | 50.892561 |
| rf100／V25 best37 | 9.626736 | 12.582781 | 30.463576 | 60.264901 | 42.779526 | 68.763274 |

| 数据集 | N／组 | Median %FS | p90 %FS | p95 %FS | >1／2／5%FS总数 |
|---|---|---:|---:|---:|---|
| industrial | 1395／52 | 17.464581 | 51.336344 | 58.967332 | 1285／1179／917 |
| rf100 | 151／35 | 3.402451 | 24.836764 | 41.365148 | 132／105／60 |

| 数据集／参照 | 逐图改善／恶化 | 组改善／恶化 | >1救回／新增 | >2救回／新增 | >5救回／新增 |
|---|---|---|---|---|---|
| industrial／v16 | 390／1005 | 4／48 | 71／143 | 86／181 | 102／250 |
| industrial／旧GeoPRR历史 | 428／967 | 20／32 | 84／100 | 134／162 | 96／259 |
| industrial／V24 | 814／581 | 26／26 | 109／32 | 209／74 | 305／64 |
| rf100／v16 | 76／75 | 10／25 | 10／27 | 18／28 | 24／26 |
| rf100／旧GeoPRR历史 | 62／89 | 10／25 | 11／34 | 12／35 | 17／31 |
| rf100／V24 | 78／73 | 12／23 | 18／13 | 38／18 | 43／17 |

救回为参照>阈值、本模型≤阈值，新增为反向跨越。完整同人口标量对齐，两个外域coverage=1、输入／scalar失败均0。相对v16，Industrial mean／tail升高10.285986／14.812458个百分点，>5新增250／救回102；RF mean／tail降低0.538040／1.430570个百分点，但>5新增26／救回24，max为68.763274%FS。相对旧GeoPRR，RF mean／tail升高3.019657／10.030881个百分点，>5新增31／救回17。相对V24，Industrial >5救回305／新增64；RF主体Acc5改善，但mean／tail升高1.899916／17.454251个百分点。

V24 RF mean7.726820／tail25.325275%FS相对v16的收益保持；不能为突出V25称旧方法全无收益。v18／v21／v22／v23／v12的完整同人口参照、组差异和阈值计数保留在[比较表](comparison.csv)与[指标](metrics.json)，不拼接不同checkpoint的最佳域成绩。

| 源checkpoint／人口 | Mean %FS | Tail10 %FS | Max %FS | >1／2／5 |
|---|---:|---:|---:|---|
| best_source／original | 0.427194 | 2.031386 | 58.511108 | 96／23／8 |
| best_source／roll | 0.785448 | 3.387103 | 17.675227 | 106／30／6 |
| last_source／original | 0.466680 | 2.755140 | 72.221673 | 77／25／10 |
| last_source／roll | 0.764250 | 3.337856 | 19.736266 | 95／26／5 |

best37 R=1.657783%FS，last60 R=1.830981%FS。R仍为原mean＋rollmean＋原tail10＋rolltail10四项normalized error平均，tail严格ceil10%（151／52）。60−37=23≥20，因min60约束在第60轮完整结束，未在第57轮提前停止；LR在31／46／55轮后下降未重置counter，无resume。source_training_cdf_log_score是本轮唯一CDF log-score字段，不与旧SmoothL1 loss数值排序。

| 源checkpoint | ordinary1495 mean %FS | ordinary1497 mean %FS | severe6 mean %FS |
|---|---:|---:|---:|
| best_source | 0.399747 | 0.410963 | 5.539712 |
| last_source | 0.429726 | 0.450085 | 4.659758 |

| 固定源sample | best37 error %FS | last60 error %FS |
|---|---:|---:|
| sync_3172 | 6.685105 | 28.825216 |
| sync_2810 | 0.164640 | 0.283927 |
| sync_3663 | 1.513592 | 2.689460 |
| sync_9323 | 1.155996 | 0.075734 |
| sync_8195 | 0.469553 | 0.210953 |
| sync_9 | 0.194534 | 0.231073 |
| sync_4025 | 8.536494 | 44.191593 |
| sync_338 | 12.100729 | 14.848363 |
| sync_4658 | 0.843553 | 1.385590 |
| sync_615 | 2.321637 | 1.039060 |
| sync_15913 | 1.435083 | 0.182265 |

best原源mean0.427194／tail2.031386%FS低于v16的0.526217／2.903382%FS，但max58.511108%FS（sync_12939）仍大；last max增至72.221673%FS。源拟合改善没有转化为两个外域整体改善。四份完整source best／last标量CSV保留原1506＋roll512，不补评旧模型的roll512。

| 外域人口 | N | Mean %FS | Tail10 %FS |
|---|---:|---:|---:|
| industrial／zero | 588 | 30.262932 | 62.235925 |
| industrial／nonzero | 807 | 15.931093 | 56.545336 |
| industrial／0<y≤.05 | 99 | 39.657238 | 74.448956 |
| rf100／zero | 6 | 13.377731 | 68.763274 |
| rf100／nonzero | 145 | 9.471522 | 41.047276 |
| rf100／0<y≤.05 | 19 | 29.760686 | 65.157805 |

RF100固定41／6／46／140的V25误差分别为68.249816／4.332310／2.652732／10.087143%FS；完整连续读数及旧模型参照、Industrial focus、所有group、zero／nonzero／nearzero均见metrics与标量CSV。以上是保存结果回顾，未用于本轮标签、阈值或配置选择。原14494／1506／4000、ordinary1495／1497／severe6和focus人口不变。

实际模型8762448参数／233个参数tensor／465 state，完整cached ImageNet encoder8434512参数／462 state仅在初始化逐项保留；全部CNN与新327936参数Linear1280→256 head共同源训练，BN只在source train更新、validation及外评冻结。完整semantic1280／conv_head／norm_head保留，没有V24的原encoder全程固定或原paramgrad0约束。FP16 encoder／head后以FP32 log_softmax生成256节点分布，稳定左右logcumsumexp计算唯一CDF Bernoulli log-score，部署全质量期望Σpᵢxᵢ／Σpᵢ，分母包含全部256质量。无旧task权重、adapter、query、几何分支、teacher、额外loss、clip、温度、峰选择或融合。

源完全复用V22的22686unique：old18590一次＋roll4096四次=34974呈现／547 batch，每轮验证原1506／24调用＋roll512／8调用。256LINEAR／ImageNet与原光度不变，没有新增RGB／GT／增强／名单。roll512独立layout／ID／group／声明seed但sharedHDR；原20096 metadata缺seed，旧1506scene guard保持。RGB与光度子合同同v16，完整部署模型合同不同。

source于2026-10-24T14:41:33.376549+00:00完成，Industrial于2026-10-24T14:41:33.460751+00:00启动、2026-10-24T14:41:45.230887+00:00结束；RF于2026-10-24T14:41:45.274803+00:00启动、2026-10-24T14:41:55.895579+00:00结束。两域各一次冻结同一source-best37，last只源；目标训练、总体统计拟合、BN适配、checkpoint／epoch／早停／训练超参／阈值／ROI／温度选择均0。推理只nativeROI RGB，RFscalar离线四点派生，输入／scalar失败始终按完整原图分母error1。

实际训练2098440图／32820次CNN与loss backward尝试＝32807有效更新＋13AMPskip；validation121080图／1920次CNN，外域1546图／25次CNN／0back／0update，正式合计2221066图／34765次CNN。head实际calls／images在每阶段与CNN完全相同。pref单列3CNN／8图／1combinedback／0optimizer；正式fresh重建，未沿用prefBN，见证确认encoder／stem／semantic neck／head必要梯度有限非零。

训练／验证pipeline4948.068876／247.773005秒，source process／full5237.863853／5238.786063秒；source至最后外评supervisor5262.123951秒，含pref5282.333375秒。source peak allocated／reserved2224756224／2858418176 bytes；外域pipeline与forward、process、full及峰值和旧模型不同训练预算的成本参照均单列在[成本](costs.json)。分布／CDF的额外计算不因一次CNN而视为免费，旧源生成只引用已有记录，不重复计新渲染。

完整model／optimizer／scheduler／scaler／RNG／history／totals／init与源合同的保存机制由执行代码和保存metadata支持；本次审计未加载checkpoint内容，不能将文件存在及metadata写成内容重载验证。四阶段complete／exit0、父进程退出、收集时GPU空闲。唯一独立CPU保存数字／执行核对68292项通过、0失败、最大差1.4210854715202004e-14，无模型／weights／RGB／GT／旧测试或旧审计执行；见[numeric audit](numeric_execution_audit.json)、[60轮history](history.jsonl)、[best](best_validation.json)、[last](latest_validation.json)与[显式allowlist](public_allowlist.json)。

## Limitations

训练范围、分布容量与监督共同改变，源改善和外域差异不能隔离单因素因果，也不证明所有有序监督无效。源低误差不保证外域mean／tail／max，完整期望仍可平均多峰。两域为已查看开发人口，单seed20262020不支持SD、显著性、新盲测、普适性或新理论；旧GeoPRR历史pixels／完整权重链未重新认证。全部旧模型、失败、G1／anchor／65轮等保留，未自动promote。rawrun／fullcontracts／完整源人口数组／RGB／weights／H/C/XYZ/U/M／bbox／features／privatepaths保持私有，只规范化标量工件公开。真人已授权继续独立源域结构研究，本轮完成后不延长当前配方或重复外评。
