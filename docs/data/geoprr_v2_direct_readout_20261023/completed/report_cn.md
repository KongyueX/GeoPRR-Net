# v15冻结GAP特征的独立直接读出：完整负结果

2026-10-23。**新增直接读出在源验证本身没有学好：source-best19为17.178667%FS，远差于同次前向原生几何读出的0.063613%FS；Industrial／RF100为39.015321／38.410688%FS，固定0.5输出均值同样回退。** 这不是“源域已学好、仅zero-shot外域失败”。本轮否定的是冻结feedback48的960维GAP特征＋线性sigmoid头在这一固定源预算下的收益，不能据此否定端到端直接进度路线或旧Raw模型的表示。

目标域没有用于拟合、统计量、checkpoint、epoch或超参数选择。这里zero-shot仅指0目标训练／适配；两套外域此前已查看，属于开发人口，不能称全新盲测。三个readout在每张图的同一次CNN前向中产生，不能按三个模型重复计算CNN成本。

## 同人口主要结果

NMAE单位为%FS，越低越好。新head仅按源1506 direct NMAE从完整20轮中选出最早best19；geometry原生模型始终是冻结epoch48，head的19不是它的新训练轮次。half固定使用同一个source-best19 head，未单独选epoch。

| 同一次CNN前向的readout | 源1506 | Industrial1395 | RF100151 |
|---|---:|---:|---:|
| geometry | 0.063613 | 17.862102 | 10.938938 |
| direct | 17.178667 | 39.015321 | 38.410688 |
| fixed_half | 8.591947 | 26.078151 | 23.730463 |

固定half为两个标量预测的0.5算术均值，区别于v14的权重中点；它没有router或目标域自适应。仅冻结外评source-best19，last20没有外域评分。全部源best／last和外部combined逐图保留三种读出，CSV中的target／prediction／error为归一化量程分数，表格再乘100记%FS。

## 源普通人口、严重错误和重点样本

| head checkpoint／readout | 源NMAE | ordinary1495 | ordinary1497 | 原严重6均值 | >1／>2／>5数 | max及ID |
|---|---:|---:|---:|---:|---|---|
| best_source geometry | 0.063613 | 0.058148 | 0.058458 | 1.180558 | 5/2/0 | 4.558325，sync_3663 |
| best_source direct | 17.178667 | 17.113617 | 17.093159 | 28.427395 | 1435/1354/1153 | 91.853232，sync_376 |
| best_source fixed_half | 8.591947 | 8.559012 | 8.548648 | 14.430078 | 1355/1207/855 | 45.865929，sync_376 |
| last20 geometry | 0.063613 | 0.058148 | 0.058458 | 1.180558 | 5/2/0 | 4.558325，sync_3663 |
| last20 direct | 17.196081 | 17.134407 | 17.114879 | 27.967062 | 1427/1352/1150 | 91.609721，sync_376 |
| last20 fixed_half | 8.600785 | 8.569538 | 8.559640 | 14.199911 | 1353/1209/850 | 45.744174，sync_376 |

| head checkpoint／readout | 3172 | 2810 | 3663 | 9323 |
|---|---:|---:|---:|---:|
| best_source geometry | 0.094023 | 3.096378 | 4.558325 | 0.029171 |
| best_source direct | 1.427993 | 42.097986 | 12.772204 | 49.355865 |
| best_source fixed_half | 0.761008 | 22.597182 | 8.665264 | 24.663347 |
| last20 geometry | 0.094023 | 3.096378 | 4.558325 | 0.029171 |
| last20 direct | 3.061929 | 42.395681 | 14.102262 | 49.964336 |
| last20 fixed_half | 1.577976 | 22.746029 | 9.330294 | 24.967583 |

原生几何在源验证的>5%FS数量为0；direct best19新增1153例>5%FS，fixed half新增855例，均无这一阈值的救回。direct的2810／9323误差分别为42.097986／49.355865%FS，读出在源域就存在大错，不将其解释为仅外域过拟合。普通1495／1497、严重6及focus名单保持不变，未为失败探针修改人口。

| best19相对同次geometry | 源逐图改善／恶化 | 源组均值改善／恶化 | >1救回／新增数 | >2救回／新增数 | >5救回／新增数 |
|---|---|---|---|---|---|
| direct | 3/1503 | 0/70 | 0/1430 | 0/1352 | 0/1153 |
| fixed_half | 4/1502 | 0/70 | 0/1350 | 0/1205 | 0/855 |

完整逐图连续效果、救回／新增ID、普通分位数、全部组和last20对应结果见[完整指标](metrics.json)，不只保存总均值。

## 外域准确率、零位与尾部

| 数据集／readout | Acc@1／2／5%FS（%） | 最差10%均值／max %FS | 零位／非零NMAE %FS | 正近零NMAE %FS |
|---|---|---|---|---:|
| industrial geometry | 7.741935/13.261649/27.168459 | 54.629123/88.554893 | 23.313352/13.890188 | 24.118316 |
| industrial direct | 1.075269/2.867384/7.813620 | 91.713947/99.990165 | 47.814528/32.604002 | 45.138056 |
| industrial fixed_half | 3.082437/5.161290/12.473118 | 65.307806/91.711229 | 35.563940/19.166572 | 34.443832 |
| rf100 geometry | 32.450331/55.629139/66.887417 | 51.643896/65.527999 | 0.557851/11.368500 | 23.671173 |
| rf100 direct | 1.324503/2.649007/7.947020 | 87.318192/95.262012 | 16.355216/39.323328 | 68.784640 |
| rf100 fixed_half | 1.986755/2.649007/11.920530 | 62.200542/75.948116 | 8.456533/24.362488 | 46.111226 |

Industrial零位／非零为588／807、正近零99；RF100为6／145、正近零19。zero按原生FP32 target==0，正近零为0<target≤0.05，另保存中间0.05<target<0.95。两个外域均无0.95≤target≤1样本，不给出空人口的满量程成绩；RF100零位只有6张，四点离线派生标量的来源照实披露。

| 相对同次geometry | 数据集 | ΔNMAE百分点 | 逐图改善／恶化 | 组均值改善／恶化 | >5救回／新增数 |
|---|---|---:|---|---|---|
| direct | industrial | +21.153218 | 333/1062 | 3/49 | 80/350 |
| fixed_half | industrial | +8.216049 | 453/942 | 19/33 | 115/320 |
| direct | rf100 | +27.471750 | 17/134 | 1/34 | 3/92 |
| fixed_half | rf100 | +12.791525 | 21/130 | 2/33 | 1/84 |

## 冻结表示、train-only统计与执行证据

使用完整v6 feedback48原生模型，496 state、7461060参数、global模式。CNN、decoder、几何角色头与所有BN buffers在eval下保持不变；保存的source cache和两套外评阶段均逐项比较496 state且一致。只读hook在同一前向取实际末层特征，经GAP得到960维，不修改geometry输出。

新增头为sigmoid(linear((f-mean)/max(std,1e-6)))，weight／bias从0开始，共961个可训练参数。mean／std仅由18590×960源训练缓存计算，std为unbiased=False；未混入1506验证或任何外域特征。独立CPU proof确认保存mean／std与源训练精确一致、与验证及train+validation均值不同；用保存的w／b、source stats和验证特征独立执行F.linear+sigmoid，source-best19预测与保存表最大差严格0。该检查不加载native CNN、不运行CNN、不反传或优化，见[统计量与读出证据](source_statistics_proof.json)。

源数据为已有端点替换18590 unique，原验证1506、ID／group分离，512端点替换与既有验证不变。cache使用augment=False、原768／192、ImageNet／AMP、train batch32、val batch16、workers4。新head仅用源归一化读数监督，CPU固定20轮、batch256、seed20262020、AdamW LR1e-3／weight_decay1e-4、余弦min1e-5、normalized L1；73批／轮，source-best仅按完整1506 direct NMAE选择，平分取早轮。source stats、epoch和参数均无Industrial或RF100拟合。

## 真实成本

| 阶段 | 队列wall秒 | 保存完整process秒 | 新CNN图像／调用 | 训练反传／更新 |
|---|---:|---|---|---|
| 源train＋validation cache | 153.621350 | 152.279737 | 20096/676 | 0/0（CNN） |
| 20轮CPU head | 7.320929 | 6.873959 | 0/0 | 1460/1460（仅CPU head） |
| Industrial三readout同次前向 | 18.650663 | 17.553659 | 1395/88 | 0/0（CNN） |
| RF100三readout同次前向 | 13.803992 | 12.733593 | 151/10 | 0/0（CNN） |

正式CNN合计21642张／774次调用：源train18590／581calls、源validation1506／95calls，外部1546／98calls；CNN backward／optimizer均为0。CPU head实际20轮、371800次源训练特征呈现、1460次反传及1460次optimizer更新。因此只能说CNN没有训练，不能将整个实验写成0反传／0更新。head训练核心累计0.802432秒；7.320929秒为fit阶段队列wall，包含加载与进程边界。完整正式队列wall193.554400秒。

必要运行见证单独保存：1次CNN调用／2张真实源图、1次CPU head反传、0 optimizer，不计入正式21642／774成本。统计proof另用0.243250秒CPU，0 CNN、0反传／更新，不混入正式训练预算。缓存是一次unique、无光度增强的特征提取，与旧709批donor／recipient全量微调曝光不同，不冒称匹配微调控制。

外评仅source-best19，原生ROI、768／192、ImageNet、eval／AMP、batch16／workers4，head只接收RGB产生的特征。完整1395／151及normalized error1规则保留，实际全部输入与读出有效；没有目标BN、统计量、ROI、温度、阈值、TTA或标签修正。全部三个读数共用同一次CNN前向，无新增渲染。

完成后的一次独立CPU数值／执行核对142946项全部通过、失败0、最大数值差1.42e-14，覆盖3012条source combined行、1546条external combined行、每行三读出、history选择、真实计数及train-only统计proof。未重载native权重、执行新GPU、旧GT或旧测试。归档仅复用最终已完成核对，没有新的评分循环。

## 与旧GeoPRR历史优势的关系

旧GeoPRR clean同人口／ROI谱系的历史均值仍较低，但当前没有完整旧权重链，执行时像素完全一致也未重新认证。历史内部诊断为：

| 外域 | Raw base | polar | relational | 最终输出 |
|---|---:|---:|---:|---:|
| Industrial1395 | 11.642999 | 11.741234 | 11.642999 | 11.643618 |
| RF100151 | 6.569962 | 6.696136 | 6.569962 | 6.607079 |

这两套clean人口全部行relation_invalid，clean入口禁用SARN关系，relational等于base，最终融合均值也未优于base；几乎全部优势已经存在于Raw base标量读出，不能直接记给SARN、polar或router。旧base是经直接进度目标训练的Raw表示，在末层图像特征上GAP、线性projection＋sigmoid，再构造保持该均值的posterior；把一个新线性层接到已几何专门化的冻结表示，不等同于恢复其端到端Raw路线。

旧主模型Raw实际为12818候选训练图、60轮、12058有效更新，其源池16000中另3182用于router；独立EfficientNet全16000 baseline是不同训练合同，不能混作旧GeoPRR锚点。旧loader单调校准默认identity，尚无历史主成绩使用knots的记录，不用未知校准解释优势。细节与证据链接见[直接读出协议](../../../GEOPRR_V2_DIRECT_READOUT_ZERO_SHOT_PROTOCOL_CN.md)。

## 结论与下一独立方案

结束这一个冻结GAP线性头探针，保留source-best19、last20和全部负结果，不机械延长20轮。结果只表明当前冻结特征＋小线性头未在源上学好；它不支持“已学好源域但仅在zero-shot翻车”的解释，也不否定完整语义表示与端到端直接读数训练。

用户最新要求继续0目标监督的泛化迭代，不限于旧GeoPRR。独立v16使用既有ImageNet缓存的完整MobileNetV4语义encoder，恢复features_only丢弃的conv_head960→1280和norm_head共6个state／1231360参数；离线CPU构造确认去分类器encoder8434512参数／462state、实际embedding1280，新增1281参数point head后8435793参数／464state。它是新源域256 RGB、60轮、SmoothL1(beta0.05)端到端训练合同，不是延长v15，也不使用Industrial／RF100拟合。4项CPU检查与4张真实源RGB运行见证通过后，v16已于北京时间2026-10-23 12:19:44启动正式源训练；两外域评估尚未开始，没有新外域性能结论。见[完整语义方案协议](../../../GEOPRR_V2_FULL_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)。

## Limitations

本轮是冻结feedback48 GAP affine／sigmoid容量探针，不是端到端旧Raw路线的复现；线性标准化不增加函数容量，20轮缓存预算也不同于完整微调。两外域已被查看并启发设计，0目标训练不等于从未查看的盲测，不声称显著性或新算法认证。旧GeoPRR历史像素与私有权重链当前不可重放，训练人口／轮次差异不能归因为单一骨干或读出机制。源中间各epoch有history摘要，但不保存每轮逐图；source-best19与last20逐图保留。RF100标量由离线四点派生、零位只有6张。原模型、所有旧失败／负结果、14494／1506／4000与冻结普通／严重名单保留；RGB、缓存特征／统计向量、head／native权重、raw H／data_contract及private trace均不公开。
