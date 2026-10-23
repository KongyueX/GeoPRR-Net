# v16完整语义MobileNetV4直接进度：完整结果

2026-10-23。**完整语义通路的端到端源训练将Industrial NMAE降至11.686054%FS，接近旧GeoPRR历史11.643618%FS但仍更高；RF100为10.164776%FS，未达到旧GeoPRR的6.607079%FS或v12 local端点版的7.659157%FS。** 源1506最优为第59轮，NMAE0.526217%FS、max65.953931%FS，>1／2／5%FS为127／40／13。保留该Industrial收益与源／RF100代价，没有全面最优模型。

本轮完整60轮及冻结best59的两次外评已全部结束，无resume。所有Industrial／RF100训练、统计量拟合、checkpoint和超参数选择均为0；源best冻结后才完成两套外评。独立[v17空间残差读出协议](../../../GEOPRR_V2_SEMANTIC_SPATIAL_ZERO_SHOT_PROTOCOL_CN.md)已冻结，源训练与结果以独立记录为准。

## 同人口主要结果

误差单位为NMAE %FS，越低越好。旧结果与当前768几何模型作为保存参照，新模型声明256输入。

| 模型／checkpoint | 源1506 | Industrial1395 | RF100151 | 源max |
|---|---:|---:|---:|---:|
| historical_old_geoprr_clean_source_only | — | 11.643618 | 6.607079 | — |
| v15_same_pass_native_feedback48 | 0.063613 | 17.862102 | 10.938938 | 4.558325 |
| v11_feedback_endpoint_best1 | 0.068488 | 17.353598 | 9.643770 | 9.506392 |
| v12_local_endpoint_best1 | 0.083392 | 20.539125 | 7.659157 | 18.677585 |
| v16_best_source | 0.526217 | 11.686054 | 10.164776 | 65.953931 |
| v16_last60 | 0.541737 | — | — | 66.018966 |

相对v11 feedback端点best1，Industrial均值降低32.6592%，899张改善、496张恶化，38／52组均值改善；相对v12 local端点best1降低43.1034%。相对旧GeoPRR历史均值仍高0.042436个百分点，逐图685张改善、710张恶化。RF100相对v11／v12端点版分别高0.521006／2.505619个百分点，相对旧历史高3.557697个百分点。

v15冻结GAP线性探针source-best为17.178667%FS，本轮完整端到端语义读出达到0.526217%FS；这条直接读数路线能够取得远好于该小头探针的源结果，但原几何读出的0.063613%FS仍更低。两套外域仅评估source-best59，last60只保留源验证，不以目标标签选择epoch。

## 外域连续逐图、组与准确率

Δ为本轮减去参考模型的NMAE百分点，负值为改善。

| 数据集 | 参考 | ΔNMAE百分点 | 逐图改善／恶化 | 组均值改善／恶化 | >5%FS救回／新增数 |
|---|---|---:|---|---|---|
| industrial | v11_feedback_endpoint_best1 | -5.667543 | 899/496 | 38/14 | 341/105 |
| industrial | v12_local_endpoint_best1 | -8.853071 | 966/429 | 39/13 | 322/73 |
| industrial | historical_old_geoprr_clean_source_only | +0.042436 | 685/710 | 32/20 | 154/169 |
| rf100 | v11_feedback_endpoint_best1 | +0.521006 | 78/73 | 24/11 | 20/31 |
| rf100 | v12_local_endpoint_best1 | +2.505619 | 65/86 | 18/17 | 18/31 |
| rf100 | historical_old_geoprr_clean_source_only | +3.557697 | 58/93 | 15/20 | 15/27 |

| 数据集 | Acc@1／2／5%FS（%） | median／p90 %FS | 最差10%均值／max %FS |
|---|---|---|---|
| industrial | 13.046595/22.293907/44.874552 | 6.489011/29.755181 | 45.734073/86.481965 |
| rf100 | 23.841060/37.086093/61.589404 | 3.367483/36.972770 | 44.210096/54.413070 |

完整[指标](metrics.json)、两份groups.csv和外评summary保留每个组／component、所有>1／2／5%FS救回和新增ID、连续误差差值及分位数。Industrial与RF100的收益方向不同，不能用Industrial接近旧均值代替全部外域判断。

## 源普通人口、严重错误与重点样本

| 源checkpoint | ordinary1495 | ordinary1497 | 原严重6均值 | >1／>2／>5数 | max及ID |
|---|---:|---:|---:|---|---|
| best_source | 0.492317 | 0.507310 | 6.043201 | 127/40/13 | 65.953931，sync_8195 |
| last60 | 0.506435 | 0.521481 | 6.285046 | 136/45/13 | 66.018966，sync_8195 |

| 源checkpoint | ordinary1495 median／p90 | 源全量最差10%均值 |
|---|---|---:|
| best_source | 0.234938/0.870437 | 2.903382 |
| last60 | 0.254455/0.911062 | 2.915384 |

| 源checkpoint | 3172 | 2810 | 3663 | 9323 | 8195 |
|---|---:|---:|---:|---:|---:|
| best_source | 15.960631 | 0.145596 | 8.968988 | 0.066400 | 65.953931 |
| last60 | 16.211846 | 0.170958 | 9.430959 | 0.108570 | 66.018966 |

best59相对v11 feedback端点best1在>5%FS救回1张、新增13张；相对v12 local端点best1无救回、新增12张。源70个组均值相对这两个端点参照均更高。2810与9323读数较准，但3172、3663和新max8195仍是明显代价。源全部1506、普通1495／1497、严重6与历史名单保持不变，完整源best59／last60逐图公开。

## 零位、非零和典型RF100差异

| 数据集 | 零位／非零NMAE %FS | 正近零NMAE %FS | 中间范围NMAE %FS |
|---|---|---:|---:|
| industrial | 14.150883/9.890119 | 29.390073 | 7.163431 |
| rf100 | 5.169570/10.371474 | 6.850400 | 10.902430 |

Industrial零位588／非零807、正近零99、中间708；RF100零位6／非零145、正近零19、中间126。zero使用原生FP32 target==0，正近零为0<target≤0.05，中间为0.05<target<0.95。两个外域都没有0.95≤target≤1样本，空人口成绩不填造。

| RF100样本 | target | v16预测 | v16误差 %FS | v12 local误差 %FS | 旧GeoPRR误差 %FS |
|---|---:|---:|---:|---:|---:|
| rf100_test_000046 | 0.135899 | 0.680029 | 54.413070 | 0.598595 | 2.433078 |
| rf100_test_000063 | 0.143340 | 0.630799 | 48.745841 | 18.341178 | 0.746438 |
| rf100_test_000007 | 0.128617 | 0.590130 | 46.151325 | 14.879149 | 0.819263 |
| rf100_test_000006 | 0.268190 | 0.728362 | 46.017158 | 0.800541 | 30.195218 |
| rf100_test_000072 | 0.413331 | 0.411998 | 0.133330 | 40.639678 | 0.389096 |
| rf100_test_000092 | 0.399042 | 0.397821 | 0.122091 | 38.764992 | 1.290172 |

v16救回了部分几何模型的大错，例如072／092；也把046等低进度图预测到高进度，形成新的大错。这些保存案例和逐图相反方向说明模型间有实际差异。当前结果没有实现新的融合、router或下一结构；完整15个代表案例及成对计数保存在指标中。

## 模型、输入和训练选择

完整encoder来自同一已有ImageNet缓存，保留global pool／conv_head960→1280／norm_head；462个encoder state、8434512参数，新Linear(1280,1)＋sigmoid为1281参数／2 state，合计8435793参数／464 state。新增point以seed20262020、std0.001、bias0初始化，未使用旧任务checkpoint或teacher。encoder的num_features字段960表示neck输入，实际语义embedding1280。

源数据为既有端点替换18590 unique、原验证1506，训练1679组／验证70组且ID／group分离。ScalarRGBDataset仅计算归一化scalar y，require_geometry=False，无donor／recipient配对；原bbox只用于ROI准备，H／field／mask／虚拟角色不输入模型或loss。256 RGB直接LINEAR resize、ImageNet归一化，保留原光度增强：contrast／saturation U[0.8,1.2]、brightness U[-0.1,0.1]。

全模型源训练60完整epoch，batch64、val64、workers4、AMP、fresh AdamW、weight_decay1e-4、gradient_clip5；唯一监督SmoothL1(pred,y,beta0.05)。每轮18590图／291批和1506验证／24批。两段30轮余弦LR按冻结配方执行，epoch1／30／31／60分别为3e-4／8.21715695e-7／3e-5／8.21715695e-8。源1506 NMAE选最早最优best59，末轮last60单独保存。

## 实际成本与完成记录

| 阶段 | CNN图像 | CNN调用 | backward | 有效optimizer／AMP skip | 核心wall秒 |
|---|---:|---:|---:|---|---:|
| 60轮源训练 | 1115400 | 17460 | 17460 | 17454/6 | 2761.025849 |
| 60轮源验证 | 90360 | 1440 | 0 | 0/0 | 183.317439 |
| industrial | 1395 | 22 | 0 | 0/0 | 8.747997（pipeline） |
| rf100 | 151 | 3 | 0 | 0/0 | 7.713571（pipeline） |

正式合计1207306张CNN图像／18925次调用；17460次源反传、17454有效更新、6 AMP skip，外域反传／更新均0，无resume。训练阶段队列wall为2973.988755秒，Industrial／RF100为12.111239／10.986934秒，controller完整wall2997.258767秒。训练完整process2973.163637秒，峰值CUDA allocated／reserved为2218448384／2837446656bytes。

外评完整process分别11.406062／10.302606秒，batch64、workers4、eval／AMP，source-best冻结后每域一次。所有输入与预测有效，完整1506／1395／151分母及normalized error1规则保留。新256实际输入见证1次CNN／4张源图／1次反传／0optimizer单独记录，不混入正式计数；本轮新增渲染为0。

一次独立CPU数值与执行核对81479项全部通过、失败0、最大数值差1.42e-14，核对完整60轮、LR边界、best选择、真实计数、3012条保存源预测、1546条外部预测及组／component汇总。归档只使用已经完成的核对与metadata，没有新模型、GPU、旧GT或测试循环。见[数值执行核对](numeric_execution_audit.json)、[60轮history](history.jsonl)、[初始化记录](initialization_record.json)和[完成记录](completed_record.json)。

## 旧GeoPRR结果所对应的路径

旧clean保存诊断中，两套人口全部relation_invalid、clean禁用SARN关系，Raw base与relational相同；其Industrial／RF100 base为11.642999／6.569962%FS，最终为11.643618／6.607079%FS。最终融合没有优于base，旧clean优势不能直接归给SARN、polar或router。旧Raw以直接进度目标训练其图像表示；v16检验新的完整语义直接模型，保留与该方向的联系。

## 本阶段结论

完整语义端到端直接读数远优于v15冻结线性小头，并把Industrial推进到接近旧GeoPRR历史均值；仍未在RF100或源严重错误上形成全面优势。保留best59、last60、原模型和全部负结果，本阶段已完整结束。独立[v17空间残差读出协议](../../../GEOPRR_V2_SEMANTIC_SPATIAL_ZERO_SHOT_PROTOCOL_CN.md)已冻结，源训练与结果以独立记录为准。

## Limitations

- 本轮同时改变完整ImageNet初始化历史、语义neck、scalar目标、256输入与60轮预算；不能将收益单独归因于neck或骨干。768几何模型使用不同网络张量、目标和曝光，同ROI人口不等于输入identity。
- 旧GeoPRR历史同人口／ROI谱系和target记录保留，但完整历史权重链与执行像素尚未重新认证。旧主Raw为12818候选训练图／60轮，另3182用于router；独立EfficientNet全16000 baseline不能混作旧任务锚点。
- Industrial与RF100已被查看并参与开发观察；zero-shot仅表示0目标训练、统计量拟合、BN适配、checkpoint或超参数选择，不能称新盲测、统计显著性或普适泛化。RF100 scalar由离线四点派生、zero仅6张。
- 指标中的GT辅助逐图最小误差oracle仅描述保存预测的互补性，不是实现的router、融合性能或未来可达成绩。独立[v17空间残差读出协议](../../../GEOPRR_V2_SEMANTIC_SPATIAL_ZERO_SHOT_PROTOCOL_CN.md)已冻结，源训练与结果以独立记录为准。
- 单seed，intermediate训练loss仅按保存SmoothL1(beta0.05)摘要检查有限，未从验证预测重建训练loss。当前外评只source-best59，不扩seed、officialtest或epoch搜索。
- 原14494／1506／4000、普通1495／1497／严重6名单、旧G1／anchor失败与配对守恒负结果保留。公开只含规范化标量、逐图、组和运行metadata；无RGB、权重、raw run／source_data_contract、bbox、完整ID人口数组、H／C／XYZ或特征。
