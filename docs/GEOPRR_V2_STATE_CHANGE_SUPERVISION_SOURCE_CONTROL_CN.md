# 状态对应的源图像变化监督控制协议

协议`geoprr_v2_state_change_supervision_source_control_v1`。新3代码和12项必要synthetic CPU检查已完成；没有全source目标构建、真实模型或GPU结果。这是有界source机制控制，不是正式V32或推广模型。

## 问题与监督范围

固定源训练quartet的四图变化能量中98.8013%集中在最高10%像素，全局逐通道affine仅解释0.09398%；该观察只支持研究局部像素变化。共同方差V对四图相同，不能把它称state特异的针体目标。新目标为每state相对四帧均值的逐像素RGB平方残差D_s；按32x32不重叠块求和到16x16，再归一化空间质量为q_s。它是额外的source image-derived监督，可能包含阴影、遮挡、反射、JPEG和其他state残影，不是物理针体/刻度/可见mask/端点/GT几何或learned teacher。原scalar y及其提供标签资格不变。

推理仍只有原nativeROI512 RGB；原完整cached ImageNet MNV4、GAP1、完整1280语义和Linear1 scalar路径保留，不输入source partner、q、模板、GT或光度校正。辅助分支只训练时执行，不修改已否决Sobel输入或混合救过，不复活旧G1/teacher/anchor/virtualQ/VDN/输出标量配对守恒。

## 模型前的完整source准入

只复用原packet的5496训练行：687 seen parent，original与roll各4state，合计1374 quartet。两view分别构建q，保持parent闭包、manifest顺序、样本、ROI和y不变；不读旧14494/原val1506，不构建ID/OOD或真实域辅助标签，不新增RGB/render/seed。

CPU准备最多900秒、5496次RGB解码；单图最多16777216 native pixels，原crop/INTER_LINEAR512/BGRtoRGB/rawF64，不做photo增强或ImageNet拟合。每quartet必须有唯一四slot/同输入尺寸与严格相同saved bbox/effective crop；不能猜warp或用GT H/keypoint投影对齐。

全部quartet均须通过先前固定的必要条件：six pair RMSE>1/255、top10% Vmass≥0.5、全局逐通道affine解释比例<0.5、finite/bounded。Affine只作source解释诊断，不校正输入。每state还须mean D_s>(1/255)^2；q有限非负、F32质量和距1≤1e-6，且与固定next-state q的total variation≥0.05。没有clip、epsilon平滑或物理y重新归一化。

任何失败都拒绝整个人口监督，保留partial及理由，source准备正常科学停止`NOT_ADMITTED_SOURCE_OBSERVATION`，不删quartet或调阈值再试、不构造模型。全部1374通过后才将server-private qcache标记完整；形状[5496,16,16] F32，payload5627904 bytes，训练身份/映射/目标和完整合同不公开。此前四图观测的decode另计，本次完整标签构建包含这些既有图身份，不称5496张新数据。

## 唯一辅助机制与损失

训练时在原encoder.global_pool前临时local hook捕获唯一[batch,960,16,16]输入，原reader.forward自身不改。新增一个zero aux_weight[1,960,1,1]，无bias、960参数，zeros构造不消耗RNG；F.conv2d产生16x16 logits，hook在finally移除，不跨forward缓存features。Eval直接原reader、aux0calls。预期learned8436753/all234trainable tensors/full465states=encoder462+point2+aux1，必须由新检查/真实pref确认，尚不称已观察。

原encoder/point AMP与scalarF32不改；auxconv跟随outerAMP，logsoftmax/KL/q在F32。固定loss为原mean SmoothL1beta.05加`.05/log256 * meanKL(q_assigned || softmax(aux_logits))`，零质量项采用0log0。系数绑定既有beta；zeroaux初始uniform时该辅助项不超过0.05。这不是最优平衡证明，不扫lambda、target分辨率或loss/预算。

## 串行预算与对应关系反证

1. CPU准备全部source，失败即无模型停止。
2. 一次真实source0opt pref，随后fresh正确对应arm。
3. 正确arm固定15E/last15；前14轮无验证、scheduler、best或延长。每E5496 once/86batch64末56；fresh AdamW3e-4/wd1e-4/clip5/defaultGradScaler/seed20262020/workers4/loaderseed+epoch，原contrast/saturation0.8..1.2、brightness±0.1，无空间增强。q来自原图但像素坐标随上述photo不变。
4. 最终native292通过原constant双学习门槛及identity15 native联合门槛后，才同last15对同292 fixed rawRGswap一次。失败则不跑CF或后续错配arm，未运行结果null/0calls。
5. 只有正确arm完整native+CF通过，才fresh循环错配arm15E，唯一变化是同quartet给state s分配q_(s+1mod4)，scalar y/RGB均正确。结构、初始化、source人口/顺序、photo、优化器、seed和完整预算相同，不继承正确arm权重。错配arm固定末native和CF各计分一次，即使其native学习较差；这是预先负对照计分，不是模型选择或额外旧模型评估。

每arm训练82440图/1290encoder-point-aux calls及1290combined backattempt=updates+AMPskip。正确native失败总82732图/1295CNN-pointcalls；CF完成83024图/1300。两arm执行则166048图/2600CNN-pointcalls、aux164880/2580、2580combinedbacks。CPU准备、pref和数据I/O另计；正常AMPskip保留，不调scale或补更新。15E为已知原RGB可学习的有界机制预算，不称收敛保证或不限资源的优越性。

## 联合接受与停止

Identity15 native参考mean/tail/max为0.025431218243695233/0.11800955331418664/0.5379703640937805，>5为16，<=1%准确102；CF为0.026353762864682596/0.12945961876345488/0.5424728989601135，>5为17，其自身准确104。

正确arm先原constant0.5 mean和ceil30 tail各半学习gate；native须mean≤reference、tail严格低、max≤、>5≤16、原准确102无新>5；CF须mean/tail/max≤reference、>5≤17、其自身准确104无新>5。所有条件同时满足，旧color drift阈值仅诊断。

完整两arm后，正确arm在native和CF都须相对错配arm满足mean≤、tail严格低、max≤、>5≤，且不能将该view中错配arm自身<=1%准确图变成>5。两个view准确集合分别推导和报告，不能套用native人数或挑checkpoint。仅全部identityjoint与对应关系joint都过才`STATE_CHANGE_SOURCE_CONTROL_PASSED`；否则`NOT_RETAINED`或更早source监督未准入。保留所有逐图/组改善恶化、三阈值新增救回、Acc/分位/尾部/max和成本，未运行不编数。

## 必要检查与执行边界

只跑新增q/16grid/退化/TV/cycle/KLF32-gradient/RNG/toyreader-hook train-eval/计数/阶段/strict恢复CPU检查，0realMNV4/forward/weights/RGB/GPU，不重复旧variance/GT/G0/selector/tests。真实pref只一次：固定source原quartet训练4图/1CNN/1aux/1combinedback，再既有ID两batch各2，合计3CNN-point/8unique/1aux4maps/0optimizer；另同4ID CPU identity解码计入wall，总decode12不是8。见证cached462/234params0opt不变/465reload/77BNsource+1eval0、stemneckpointaux梯度存在finite非零、96016grid/1280语义、F16point及F32scalar/KL、q对应地址。初始uniform时正确/循环batchKL可能相同，不能要求虚假的初始loss差。正式两arm完全fresh，不带pref状态。

Source准备、pref、每arm所有计数/阶段/fullwall/峰内存/target存储/aux开销实记，未测point time为null。严格新protocol/config/source/aux-target/model+aux/optimizer/scaler/RNG/history/totals/completeepoch恢复，旧464或其他版本不兼容；partial/FileExists/CSV无summary保护不绕、无第二forward。

代码、协议、配置和必要NEWCPU evidence先existingmain冻结/scoped部署，actual旧父子gone/GPUidle后唯一串行。wholecomplete（包括模型前source科学拒绝）且actual父子gone/GPUempty后才ONEfresh非作者saved-number/execution CPU核对、explicitallowlist规范化归档。审计不重开weights/RGB或重算source辅助标签，相关标签来源验证边界必须如实保留。作者不能审计，不平行二审。

## Limitations

既有prepared.json包含完整source contract及嵌套train/ID/OOD人口；准备阶段反序列化并私有保留该合同，q构建仅使用train manifest行，不读取ID/OOD图像或利用其字段构建aux目标。不能把整个metadata文件parse称作严格仅train字段。完整合同/数组不会stdout或公开。

这新增source图像派生监督，不能冒称原scalar-only、物理mask或历史pixel/fullweight重新认证；C3原149/148/1 raw-label资格不改。单seed、反复使用的source开发集及人工CF不构成SD/显著性/newblindtest/真实域泛化或因果理论；对应关系控制仅支持该预算下的source解释。Source03和真实Industrial/RF不构建监督、不做模型调用/选择，外域不能训练、无标签adapt、总体statfit、BNadapt或训练/ROI/threshold/温度选择。不新weights/seed/资产/render/GT物理标签/买算力，不改系统PythonCUDAconda其他项目、不清旧资产、不碰officialtest4000、无第三方消息、不创建branch。正结果也不自动promotion或正式长训；未来正式模型须另立source预算/R/stop后同best两真实域各一次，再唯一核对。
