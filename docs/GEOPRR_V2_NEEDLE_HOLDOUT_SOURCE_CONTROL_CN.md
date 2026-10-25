# 可追溯源针型留出：五轮有界控制

本控制保持V26的完整MobileNetV4／GAP1280／单scalar结构，只检验两个问题：在固定小预算内是否能学会已见针型；若能，同一last5对一种训练未见的既有合成针型误差如何。它是源问题定位控制，不是正式V32、泛化候选或外域成绩。针型按source metadata中的canonical selected_pointer字典序留出最后一种，不用预测、Industrial或RF选择。

## 固定人口与计算预算

已记录元数据确认original1024与roll1024各四state、parent闭包和针型身份可对应，source验证128layout恰共享3种针型。训练只用另外两种的687个完整parent：2748 original＋2748 roll=5496 unique，每epoch各一次，不沿用完整V26的roll四倍重复。旧14494无因素记录训练行、原1506验证和留出种2696训练图均明确不纳入；不改变旧split、像素、标签或历史分母。

source-ID为现有roll验证中两种已见针型73layout／292图，source-OOD为留出种55layout／220图。每layout四state完整保留。完整候选名单和source arrays只保存在服务器；公开仅metadata、counts和bool。训练与两验证的layout／sample不交；不称font／HDR新域或完整22686人口结果。

训练固定5个完整epoch：每轮5496图／86批64末56，总27480图／430次CNN与backward尝试，updates＋AMPskip=430。只用last5，前四轮不验证，固定LR无scheduler／best选择／早停／延长。最后仅source-ID一次292图／5calls；学习门槛未过即结束，不读OOD RGB。通过才同last5 source-OOD一次220图／4calls。正式总量分别27772／27992图、435／439次CNN，point计数一致，两个真实外域的images／calls／backward／updates均0。V26实录线性外推训练约133秒，初始化、pref和I/O另计；不保证收敛或运行时间。

## 表示、优化和学习门槛

fresh既有缓存ImageNet MNV4 num_classes0、完整conv_head／norm_head／GAP1／semantic1280，新Linear1280→1 std.001／bias0，FP32sigmoid唯一scalar。全部CNN与point源训练，77BN仅训练源更新、eval冻结；预期8435793参数／233参数tensor／464state。不改head／pool／norm，不用GN、stylemix、二阶head、teacher、几何输入、adapter、融合或TTA，不加载旧任务权重或下载新权重。

同V26 nativeROI RGB→LINEAR512／ImageNet normalize，output128仅API、geometryFalse／pairNone；contrast和saturation0.8–1.2、brightness±0.1。freshAdamW3e-4／wd1e-4／clip5／SmoothL1beta.05、FP16 outerAMP／FP32scalar／fresh defaultGradScaler、seed20262020，train／val batch64、workers4。此五轮控制固定LR，不计算完整V26的R或进行Plateau选择。

唯一门槛在同一292行source-ID上定义：固定constant0.5与模型分别按已有归一化scalar标签计算完整mean和tail10（ceil30）。模型输入或scalar失败按error1保留292分母，合法0／1输出有效。只有model mean≤constant mean的一半且model tail≤constant tail的一半同时成立，才算本控制先建立有效源学习。未过则记录learning_gate_failed、跳过OOD，不改epoch／LR／seed／精度／针型／名单／阈值重试；通过后只报告源OOD原始指标，不用于本控制训练或checkpoint选择。

## 必要检查与执行边界

只新增CPU selector／parent闭包／固定针型排除／新曝光计数／学习门槛和OOD条件执行检查，不重复旧CPU模型构造、GT、数学或测试。精确source manifests须先一次物化并冻结于独立server目录；为物化既定名单而必要读取source metadata不算新独立验证，不重新选择针型。

一次新subset零更新pref单列3CNN／8图／1combinedback／0optimizer：4训练图来自两种已见针型各首个parent的slot0 original／roll，两次eval各2图只用source-ID，pref严禁OOD RGB或推理。见证实际512、完整semantic1280／pointF16／scalarF32、cached462state初始化保持、全部233参数0opt不变、必要stem／neck／point梯度有限非零、77BN source+1／eval额外0、464strictreload及ID输入pipeline identity。pref固定loss128再grad/128，正式fresh重建默认GradScaler及初始化，不带prefBN／状态。

code／protocol／config／NEWCPU证据先推existing main，实际旧父子退出且GPU空后才唯一pref→fresh五轮control。保存model／optimizer／scaler／RNG／history／counts／init／新protocol与source合同，失败保留partial不fresh覆盖或自动延长；不恢复或重跑旧实验。实现者不得自行启动GPU。完成后只核对本控制保存标量与执行记录，不补评旧模型或两个真实外域，不自动开长训练。

## 完成记录

固定5轮完整结束，430次backward尝试＝430更新＋0AMPskip，只用last5；最后source-ID292一次冻结评估。模型mean6.132800787730%FS≤预声明上限12.504463703047%FS，tail10为25.669996334861%FS>上限23.692208052865%FS；constant0.5 mean／tail为25.008927406093／47.384416105730%FS。均值已学会改善，但尾部未同时满足一半条件，正常completion_reason为learning_gate_failed。source-OOD RGB／forward0、metrics null、无预测文件；heldout针型问题未评估，Industrial／RF全部images／calls／backward／updates0，没有外域结果。

正式training27480图／430 CNN、source-ID292图／5调用，合计27772图／435 CNN，point一致；source-ID-only pref3CNN／8图／1back／0optimizer单列，未读取OOD RGB。表示仍为完整MNV4／GAP1280／新scalar point、8435793参数／233参数tensor／464state、77BN源更新／eval冻结。cached462初始化、零更新参数保持、464 strictreload及必要梯度只引用保存见证与执行metadata，未fresh检查checkpoint或cached权重内容。无best选择、scheduler、延长或重试，保持5E预算及原双门槛。

唯一独立CPU保存标量／执行核对49／49 PASS、0失败，结论PASS_WITH_QUALIFICATIONS；四项qualification在[审计](data/geoprr_v2_needle_holdout_source_control_20261025/completed/numeric_execution_audit.json)完整保留。正式supervisor stage wall173.437898秒、pref17.985979秒，两stage和191.423877秒，pref起点至control结束191.527038秒。原helper process171.235725361854／recorded full171.392929736525秒保留，但新entry先lazy import torch再导入计时helper，旧full_process边界标签不是新entry完整import边界；完整child-stage elapsed采用supervisor wall，point独立计时null，不填0。两stage complete／exit0，13:52:29.672489保存观察时父413065／pref413069／control413106退出、GPU空闲。

[报告](data/geoprr_v2_needle_holdout_source_control_20261025/completed/report_cn.md)、[指标](data/geoprr_v2_needle_holdout_source_control_20261025/completed/metrics.json)、[gate](data/geoprr_v2_needle_holdout_source_control_20261025/completed/learning_gate.json)、完整model292／constant292 scalar CSV和73组summary／CSV已规范化归档。根7项入口＋[completed21项](data/geoprr_v2_needle_holdout_source_control_20261025/completed/public_allowlist.json)，[总allowlist](data/geoprr_v2_needle_holdout_source_control_20261025/public_allowlist.json)28项。run／completed／queue／screen仅metadata projection，不是raw full文件。本次公开只含规范化保存标量／metadata；fullcontracts、source arrays、5496训练名单、RGB、weights、geometry、features、privatepaths及commands保持私有。全部原资产保留，不扩预算、不改门槛、不外评、不自动promotion。

## Limitations

本控制只检验现有三个合成needle资产中的一个固定留出，不能证明针型是外域失败根因，也不能推断font／HDR未见、真实域收益、显著性或普适性。source-ID与OOD其他layout因素可能混杂；低预算未学会源ID时不具有可解释的OOD结论。保留全部历史失败，不把源结果或单域收益写成整体泛化成功。完整来源名单、RGB、weights、features、geometry和privatepaths保持私有。
