# 平均与batch尾部风险：固定15轮source控制

唯一问题是：不改变原生RGB、模型、数据与15轮预算，只改变source loss的聚合，能否降低source-ID尾部，同时守住mean、maximum、>5计数和原本准确样本的新大错？这是标准风险聚合的有界source检查，不是新理论或正式V32，不保证真实外域收益。

## 源依据与唯一变化

当前同语义scalar reader默认优化batch mean SmoothL1；既有完整source选点R和联合验收同时重视mean与tail。有界源码检查未发现同一per-image top-ceil10% SmoothL1已在该source reader执行，不能扩大为所有历史均未运行的证明。前一C3匹配控制不满足大错计数和maximum联合条件，已NO_MATCHED_SUPPORT_FOR_C3，不继续C3或调整其order/epoch/门槛。

[Fan等，NeurIPS2017 Average Top-k Loss](https://papers.nips.cc/paper_files/paper/2017/file/6c524f9d5d7027454a783c841250ba71-Paper.pdf)提供标准top-k聚合背景；[Zhai等，ICML2021 DORO](https://proceedings.mlr.press/v139/zhai21a.html)提醒高loss风险对异常和噪声敏感。这里不采用DORO去样本/裁剪，不指认source存在噪声，也不据论文分数选超参。

每图仍只有一个原SmoothL1(beta=.05)，新唯一objective为：

`L = 0.5 * mean(loss_i) + 0.5 * mean(top ceil(0.1*B) loss_i)`。

B64取7、末batch56取6、prefB4取1。每轮601个upweighted slots，比例不恰等于完整人口10%；全部5496图仍通过mean项参与，无hard-sample删除或重采样。权重.5与fraction.1来自原source验收的mean/tail对称优先，不进行alpha/lambda/beta搜索。训练SmoothL1与absolute-error指标不同，batch尾部也不等于完整source尾部；目标差异不是失败原因已证。

## 固定模型、数据和预算

完全保留freshcached ImageNet MNV4/GAP1/semantic1280/Linear1 std.001bias0/sourceBN/AMP与F32scalar，原5496train/292ID packet、原native512ROI/LINEAR/ImageNet/photo、seed20262020/loaderseed20262020+epoch/batch64/workers4、AdamW3e-4wd1e-4clip5/defaultfreshScaler。真实训练为RGB5496/GBR0/BRG0每E，旧nominal slots只保留相同loader bookkeeping，不是颜色augmentation。

固定15E、1290attempts，无中val/scheduler/best/延长，last15唯一。正常AMP梯度溢出按scaler skip计数，不误作loss fault，不调scale补更新。训练读取已捕获异常在既有augment=True分支raise；未发现训练坏图占位继续的路径，实际labelnoise未知，不为此重扫RGB或旧GT。

完整提供的source scalar不变，forward仅RGB，一图一次CNN，无head/pool/norm/teacher/geometry/mask/新labels/weights/seed/数据文件/配对损失。风险聚合的额外kernel/time/peak属于实际训练成本。历史source_training_smooth_l1字段若保留，必须在config/metadata明确其值为composite risk，而非primitive mean；unweighted trainingNMAE照实记录。

## source联合验收

参考是已完整匹配identity15E的native/CF保存scalar和metadata；不开旧weights、不再forward旧模型。全部292source-ID、73layouts×4行保持，error1失败分母不删，0/1有效饱和值不误判。Provided-label比较与旧C3 rawlabel派生未另认证的范围资格保留，不新GT归一化。

Native先过原constant0.5半mean/ceil30tail最小学习gate，再同时要求：mean不高于identity15；ceil30tail严格低于identity15；maximum不高于identity15；>5计数≤16；aligned identity native误差≤1%FS的源样本中，没有新增>5%FS错误。最后一项只属于本292source-ID的准确参考集，不能冒充旧ordinary1495/1497。mean/tail/max基准精确值在机器协议中冻结，不使用显示精度。

Native任一未过即正常NOT_RETAINED，CF0CNN/无结果文件，不延长。Native全过才同last15、同292label/ROI，以已固定RGswap[1,0,2]评一次：mean/tail/max分别不差identityCF，>5计数≤17；referenceCF误差≤1%中没有新增>5错误。旧color-control漂移门槛只作诊断，不决定这个loss实验；完整mean/ceil30/max drift仍保存。

只有native与CF全部joint条件成立才MEAN_TAIL_SOURCE_CONTROL_PASSED，否则NOT_RETAINED。全部Acc1/2/5、median/p90/p95、>1/2/5新错救回、image/group改善恶化及保护集人数均保存；1/2计数不新增为acceptance旋钮。源ID与人工RGswap是同布局development view，不是独立域或新盲测。Source03/Industrial/RF全0RGB/model，不影响训练/选择或continue-stop。成功也仅供root考虑下一独立正式source版本，不promotion或继续本15E。

## 新检查、执行与记账

只必要NEWCPU检验新聚合/kceil/全样本grad贡献/B1B4B56B64有限autograd、loss/context隔离和新joint/ordinary/conditionalCF/schema/objective metadata。旧14C3/8identity/operator/selector/GT/math/教师/tests不重复，0realMNV4construct/RGB/weights/GPU。

新源pref一次3CNN8uniqueRGB1combinedback0optimizer，四native train使用新objective，四nativeID；cached462保持、233参数无更新、464strictreload、77BNsource+1eval0、必需finite非零stemneckpoint梯度、512identity/dtypes；fixedloss128/grad128只用于见证。模型/BN/RNG状态不carry到formal，formal freshdefaultScaler不改。

| 范围 | 图像呈现 | CNN/point calls | backward attempts |
|---|---:|---:|---:|
| train15E | 82440 | 1290 | 1290 |
| native | 292 | 5 | 0 |
| 条件CF | 292 | 5 | 0 |
| native未过总计 | 82732 | 1295 | 1290 |
| CF完成总计 | 83024 | 1300 | 1290 |

pref另列，identity同4ID重复decode计CPUwall，不声称总decode8。每attempt实际updates+AMPskip实记；point/isolatedloss/算子独立时间未测写null。Ownentrytimer先于torch import，phase/wall/peak准确保存。新protocol/config/source/fullstate-opt/scaler/RNG/history/totals严格resume，rawhistory/checkpoint不因命名归一化失配，无latest/模型partial不fresh覆盖，CSV无summary不二forward。

先matched归档existingmain，再冻结本新3code/protocol/config/NEWCPU证据push existingmain、scoped部署3code+3snapshotmetadata、actual旧父子goneGPUidle后唯一pref→source15。活job只读不可改配方/再加GPU/提前CF。Whole2stageexit0且父子gone/GPUempty后ONEfresh独立nonauthor保存数字CPU审计，不重复旧审计/模型或GT，规范化explicitallowlist现有main归档所有正负。RGB/weights/features/fullcontracts/trainarrays/geometry/privatepaths留server，stdout先allowlist，仅metadata/bool及允许scalarCSV/groups。

## Limitations

标准ATk风险聚合不构成新颖性认证；异常和不可约高loss可能被强化，普通准确性/新大错guard不能证明无labelnoise。单seed、source-ID和同布局人工视图不支持真实外域整体成功、SD、显著性、新blindtest或理论。原source标签派生和旧GeoPRR pixel/fullweight链未新认证。全部G1/teacher/anchor/65轮/配对守恒与其他旧结果保留，不复活主线；不新seed/weights/foundation、买算力、改系统环境、清资产、碰officialtest4000或第三方消息。
