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

## Limitations

固定RGswap已在C3研究中查看，这是一次预注册的匹配source比较，不是新盲测。相同初始化的依据是factory/cache/seed与冻结代码证据，不新开过去权重验证。Source颜色压力不能覆盖完整材质、光照、sensor或仪表结构；通过不保证真实域提升。全部旧正负/G1/teacher/anchor/65轮/配对守恒负结果保留，不能复活为主线。RGB/weights/features/fullcontracts/训练arrays/geometry/privatepaths留server，collector stdout前allowlist；不改source人口/标签/ROI、不新seed/weights/foundation/算力购买/officialtest4000/第三方消息。
