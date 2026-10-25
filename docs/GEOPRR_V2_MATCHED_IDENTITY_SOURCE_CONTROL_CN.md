# 固定15轮identity输入：匹配颜色控制的source实验

本实验只问：前一C3 source控制的颜色压力风险与prediction稳定性，是否优于相同15轮预算的fresh原生RGB输入模型？它是一个新独立source对照，不延长directFT5、不加载旧训练checkpoint、不补评旧模型，不是正式V32或泛化模型promotion。

## 为什么需要这一项短控制

已完成C3 last15在source-ID292上mean/tail为2.583311489/11.098201560%FS，固定RGswap为2.558655712/10.711229617%；原native与CF门槛都通过。但旧directFT5只有5轮，15轮预算为其三倍，不能把改善归因于颜色处理。C3 native最大误差仍55.090773106%，CF >5%FS错误31张比native28张多3张。这些事实支持先排除预算混杂，不直接启动完整长训练。

C3唯一审计保留149项148PASS/1FAIL、13617数值对比maxdiff1.4210854715202004e-14。FAIL是合同缺少现成normalized_progress/y字段而不能完成该项来源核对，没有观测到label数值冲突；native/CF/旧FT5的292 labels/ID/group精确一致。此实验不重做旧审计、不重新归一化物理标签或修改历史记录，只在同一provided scalar标签与同一source合同下作匹配比较；完整原始标签派生未另认证的限制继续保留。

## 唯一删除与固定流程

保留C3模型、全部cached初始化/seed20262020、5496训练图/687parents、292source-ID/73layouts、512 nativeROI/LINEAR/ImageNet、原photo、SmoothL1 beta.05、AdamW lr3e-4 wd1e-4 clip5、defaultfresh AMP/scaler、batch64/workers4/loader seed20262020+epoch、15E/last15/no intermediate val/scheduler/best。

只删除训练中的通道变化。三个nominal schedule slots仍按`(index+epoch-1)%3`分配以保持同一数据/随机序列机制，但三个slot全部映射RGB identity；每轮实际颜色计数RGB5496/GBR0/BRG0，每raw图identity15次。记录`order_exposures=[1832,1832,1832]`只表示nominal slots，不能写成三颜色各1832。Identity直接返回原normalizedFP32 tensor，无新RNG、参数、损失、数据文件、渲染或标签。

native source-ID与RGswap采用前一C3的同一算子和门槛；source-ID constant半mean/ceil30tail学习gate及旧FT5 native mean/tail/max/>1/2/5全部guard过后，才评同last15/同292图的固定RGswap一次。未过不补CF。后续对比以既有C3保存统计及新identity统计完成，绝不再forward C3；源needle03/Industrial1395/RF151全部0RGB/model。

## 预注册的匹配判断

两模型均具有完整native与CF292结果并通过各自native学习/primary，才作匹配判断。否则记`MATCHED_CF_EVIDENCE_INCOMPLETE`，不填缺失数值、不补评。

定义两固定source视图的风险：

`R_view = (native_mean + native_ceil30_tail + CF_mean + CF_ceil30_tail)/4`。

仅当以下条件同时成立，记`MATCHED_SUPPORT_FOR_C3`：C3的R_view严格更低；两视图>5计数总和不增加；两视图最大误差不增加；C3的平均prediction drift与ceil30 drift均严格低于identity。否则记`NO_MATCHED_SUPPORT_FOR_C3`。保持所有native/CF均值、尾部、极值、Acc1/2/5和>1/2/5新增/救回，而非只报有利视图。R_view是同source布局两种人工条件的等权风险，不是独立域、真实外域均值或显著性估计。

该主判定在identity训练与结果产生前冻结。Scheduled训练图像与CNN/backattempt预算相同，actual有效updates和AMPskip仍可能不同，必须如实保留；单seed不作统计或普适因果声明。成功也只供root在归档后考虑新的独立正式source版本，不直接promotion或延长本run。

## 检查、预算与执行

只新增identity-deletion context/actual-vs-nominal mapping、CF算子保持、新protocol/config隔离和matchedrisk/max/count/drift判断分支CPU toys；已有C3的operator/inverse/exposure/gates/旧GT/教师/tests不重跑，不真实model/RGB/weights/GPU。新pref仍3CNN8uniqueRGB1back0opt，四train nominal slots0/1/2/0实际全RGB，ID四图，cached462/233参数无更新/464strict/77BNsource+1eval0/必需有限非零梯度/dtype/nativeidentity；正式fresh无prefcarryover。

每E5496/86calls/86backattempt，固定15E82440/1290；native未过总82732图1295CNN-pointcalls，CF完成总83024图1300calls，1290attempt=updates+AMPskip。pref另列，identity验证对相同4ID再解码的CPU工作计wall。独立point和算子时间未测写null，不填0。

root须先完成C3 qualified结果归档，再将本次新3code/publicprotocol/config/新CPU证据explicitallowlist existingmain冻结push、scoped部署3taskcode/3metadata，actual旧父子gone/GPUidle后唯一pref→freshidentity15。Snapshot metadata提前检查，launchlog/receipt在pristineout外。新protocol/config/source/fullmodel-opt/scaler/RNG/history/totals严格保存与恢复，无latest或模型partial不fresh覆盖，CSV无summary不二forward。Ownentry timer在torch import前。

whole2stagecomplete/exit0、实际父子gone/GPUempty后才ONEfresh独立non-author保存数字CPU核对并规范化归档，不重复旧C3审计、不新开权重或GT数学。元数据标签一致性按实际provided字段与保存targets检查，不制造缺字段要求；raw label派生资格不隐瞒。

## 完成记录

Fresh identity15E及协议允许的native／CF一次评估完整结束，科学主判定NO_MATCHED_SUPPORT_FOR_C3。两臂均15E／82440训练呈现／1290back尝试、actual1289updates＋1skip。C3 R_view0.06737849594632048低于identity0.07481353829650483，mean／tail prediction drift也严格更低；但C3两视图>5总数59高于identity33，最大误差0.5509077310562134高于0.5424728989601135，五项未同时通过，不因有利R_view或漂移启动长训／promotion／预算或门槛扫描。

Identity native mean／tail／max2.543121824370／11.800955331419／53.797036409378%FS，>1／2／5为190／110／16；CF2.635376286468／12.945961876345／54.247289896011%FS，188／103／17。C3 native2.583311489025／11.098201560477／55.090773105621，189／120／28；CF2.558655711628／10.711229617397／53.595584630966，192／120／31。所有条件292／73layout×4state、coverage1／fail0；两臂旧native与CF门槛通过，不覆盖matched否定。Identity最大prediction漂移30.185824633%FS保留。

Identity每E实际RGB5496／GBR0／BRG0，15E总RGB82440／GBR0／BRG0；[1832,1832,1832]仅nominal slots，不能描述三颜色训练。C3→identity native图像改善150／恶化140／相同2，CF147／145／0；identity native→CF137／152／3，全部group与阈值paired计数、原scalar／pair CSV、summary／history／gates保留。Source03／Industrial／RF0RGB／model，未再forward旧C3。

正式83024图／1300 CNN及point调用，1290back＝1289更新＋1skip；pref另3CNN8unique1back0opt，ID4图额外decode CPU工作计入wall。Train446.680151045秒、eval pipeline6.870805331秒，whole supervisor482.665441178秒、pref／control supervisor19.270640597／462.370882876秒；matched-wrapper full460.987576280秒包括wrapper metadata／reference检查，排除最终completion JSON及downtime。Point／operator单独时间unknown／null，cached／BN／strict等仅保存见证与代码metadata，未新开旧weights。424199／424202／424240已退出，18:52:49 UTC保存观察GPU空。

最终唯一provided-label保存数字／执行审计171PASS／0FAIL、13726numeric、最大差1.4210854715202004e-14；C3参考统计按既有1e-12绝对容差接受末位舍入，非逐位一致。旧C3审计149checks／148PASS／1FAIL原样保留，raw-label派生范围未新增认证。见[报告](data/geoprr_v2_matched_identity_source_control_20261026/completed/report_cn.md)、[最终审计](data/geoprr_v2_matched_identity_source_control_20261026/completed/numeric_execution_audit.json)、[匹配判断](data/geoprr_v2_matched_identity_source_control_20261026/completed/matched_comparison.json)与[reference资格](data/geoprr_v2_matched_identity_source_control_20261026/completed/reference_audit_qualification.json)。根目录7项＋本独立协议1项＋[completed31项](data/geoprr_v2_matched_identity_source_control_20261026/completed/public_allowlist.json)，[总allowlist](data/geoprr_v2_matched_identity_source_control_20261026/public_allowlist.json)39项。仅规范化保存scalar／metadata公开，完整合同／训练arrays／privatepaths／RGB／weights／features／geometry保持私有，不改旧C3档案。

## Limitations

固定RGswap已在C3研究中查看，这是一次预注册的匹配source比较，不是新盲测。相同初始化的依据是factory/cache/seed与冻结代码证据，不新开过去权重验证。Source颜色压力不能覆盖完整材质、光照、sensor或仪表结构；通过不保证真实域提升。全部旧正负/G1/teacher/anchor/65轮/配对守恒负结果保留，不能复活为主线。RGB/weights/features/fullcontracts/训练arrays/geometry/privatepaths留server，collector stdout前allowlist；不改source人口/标签/ROI、不新seed/weights/foundation/算力购买/officialtest4000/第三方消息。
