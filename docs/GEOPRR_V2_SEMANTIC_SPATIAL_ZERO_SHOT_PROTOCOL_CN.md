# 完整语义表示与空间残差读出：独立源域实验

2026-10-23。状态：协议、代码及预算冻结后，正式源训练于北京时间2026-10-23 13:49:51启动，现已完整8轮并于13:57:00完成两套冻结外评，best8=last8；一次独立CPU数值／执行核对53401项全部通过。源均值小幅改善、两外域均值小幅回退，本配方不提升、不延长；[完整结果](data/geoprr_v2_semantic_spatial_20261023/completed/report_cn.md)已归档，以下保留评分前冻结配方。用户要求继续zero-shot结构迭代，Industrial与RF100不用于训练。本轮不延长v16的60轮，不改变既有模型、数据、失败记录或officialtest。

## 已有证据与可检验问题

v16完整60轮及两套冻结外评的一次独立CPU核对为81479项通过、0失败。source-best59的NMAE为0.526217%FS，Industrial为11.686054%FS，RF100为10.164776%FS。Industrial已接近旧GeoPRR历史11.643618%FS，但RF100仍差于旧模型6.607079%FS和v12 local的7.659157%FS；源验证max为65.953931%FS，不能将本轮接近一域写成全面解决。

两条读数路线存在配对互补。RF100的`meter_c_2`组14张，v16误差35.753709%FS，local为1.444202%FS；部分target约0.27被direct读成约0.72，数值上接近反向进度。另一些local误差超过30%FS的样本，direct误差小于1.5%FS。这只是方向或参考布局混淆的假设线索，尚未用图像确认机制。不能按目标标签选择分支或拟合融合。

本轮仅检验：在保留v16完整语义和直接标量通路的同时，增加一个读取局部空间证据的源训练分支，能否改善上述结构弱点。旧GeoPRR的clean优势已在Raw base出现，旧polar/router没有改善整体clean均值，所以本轮不将旧成绩作为attention有效的证明。

学习少量图像条件空间权重以汇聚局部特征，参考[TokenLearner原论文](https://proceedings.neurips.cc/paper_files/paper/2021/file/6a30e32e56fce5cf381895dfe6ca7b6f-Paper.pdf)的通用思想。该论文不证明本任务的zero-shot收益；下面的残差结构是本轮实现选择，不作为新理论声明。

## 唯一结构改动

严格加载已完成v16的source-best59完整464 state／8435793参数。其encoder、semantic neck、原point参数与全部BN buffers均冻结，始终eval。保留同一个encoder前向中的1280维semantic和原point logit，不重建`logit(clamp(prediction))`。

离线CPU实际完整模型`feature_info`已确认唯一空间入口`encoder.blocks.1`：stride8、80通道，在256输入下为32×32。通过同一次encoder前向获取该特征，不增加第二个CNN。

新增分支固定如下：

1. 1280维semantic做每样本LayerNorm后，Linear1280→96形成3个32维query；每空间位置80维特征做LayerNorm后，由1×1 Conv80→32投影。
2. 在投影特征中加入Linear2→32的规则像素中心坐标编码。坐标为RGB网格的[-1,1]位置，不是GT或预测针尖、center、H或SPD。
3. 对每个query，用独立32维score向量作用于`tanh(key+query)`，在1024个位置上softmax；分别汇聚32维投影特征和2维网格坐标，得到3×34维token。
4. 将102维token送入Linear102→32、GELU、Linear32→1，产生logit残差。最后Linear的weight和bias全零初始化，初始预测与v16父模型逐项相同；最终输出为`sigmoid(parent_logit.float()+delta.float())`。

3个token是通用可学习空间汇聚，不赋予A/P/B、针尖或真实刻度的解释。预期新增129057参数、10个state，总8564850参数、474 state，以实际构造记录为准。新分支的归一化仅使用当前样本的通道，不拟合外域总体统计。

全零输出层导致首次反传时query/key等上游梯度为零，这是预期链式法则结果；运行检查分别验证初始identity和临时非零输出层下的有限梯度，再恢复初始状态。正式训练重新加载父权重和新分支初始化，不使用运行检查后的状态。

## 源域训练与固定预算

仅用既有18590源训练图和原1506源验证，沿用v16已完成端点替换、256 RGB／LINEAR／ImageNet流程与原光度增强。`require_geometry=False`，不新增渲染、角色或几何监督，不用ROI扰动、MixStyle、teacher、额外loss或seed。

只训练新增空间分支；全部父参数不进入optimizer，CNN仅前向，BN保持不变。固定8完整epoch，train64／val64／workers4／seed20262020／AMP，fresh AdamW，weight_decay1e-4、clip5。唯一loss为源读数SmoothL1 beta0.05。LR为`1e-5+(1e-3-1e-5)*(1+cos(pi*(epoch-1)/8))/2`，不按外域效果改轮次或配方。

源选择包含epoch0父模型作为既有基准：复用v16已审计source-best59的1506逐图和NMAE，保存零残差初始checkpoint，不额外重评父模型。每轮全1506源NMAE严格降低才替换best，平局保留更早者；8轮全部执行，last8仅保存源结果。若最终best仍为epoch0，则保留父模型，不声称新增结构有效。

每轮18590训练图／291调用、1506验证图／24调用。8轮为148720训练CNN图／2328调用和head反传尝试、12048验证图／192调用；CNN反传和父参数optimizer update均为0。每轮逐项比较全部464父state及77个BN counter，记录保持不变，不用无人读取的指纹代替直接比较。有效head update与AMP skip按真实执行记录，二者之和为2328。

## 冻结外评与保存

只有8轮源训练完整结束后，source-selected best才各评Industrial1395／52组和RF100151／35组一次。若best为epoch0且残差输出层仍全零，精确复用v16两套外评作为alias，0新增CNN；否则两域1546图／25调用。正式本轮合计160768图／2520调用，或含非零epoch外评162314图／2545调用。运行检查费用另列。

两域不参与梯度、loss、特征总体统计、BN适配、checkpoint或超参数选择。推理仅接收原生ROI RGB，规则网格由输入形状生成，不用目标GT几何、target、温度、阈值或TTA。完整失败分母error1；RF100 scalar由四点离线派生，明确披露。不得以目标标签做oracle路由或搜索融合系数。

独立入口`experiments.run_geoprr_v2_semantic_spatial`，输出`runs/geoprr_generalization/iteration_v17_semantic_spatial_20261023/source_only_training`。启动前确认v16整个队列complete、父子退出、GPU实际空闲，并通过本轮唯一新源输入检查。训练保存完整model／optimizer／scaler／RNG／history／累计成本；真故障保留partial和失败日志，在同output从latest恢复，不删样本或放宽保护。

完成后收齐配置、完整history、best/last源CSV、冻结两外域逐图／分组／summary和实际成本，仅一次必要独立CPU数值与执行核对。报告每域连续误差、Acc、zero／nearzero／nonzero、组、普通1495／1497、严重6、重点3172／2810／3663／9323和新增8195、tail／max／救回新增，不按单个域宣布统一最佳。

实际启动代码c5e98a4、父412238／初始训练子412241。父子精确命令与唯一GPU进程已核实，首批64张源图loss0.00047157992958且梯度有限。见[启动记录](data/geoprr_v2_semantic_spatial_20261023/launch.json)与[零更新运行检查](data/geoprr_v2_semantic_spatial_20261023/runtime_preflight.json)；启动和小kernel不作为性能成绩。

## 限制

该假设尚未证实。只用scalar监督的attention可能选择源背景或表盘文字；冻结特征若缺乏细针信息，新head无法创造它，坐标也可能强化源布局偏好。源成绩改善不保证外域改善。两外域是已查看开发集，0目标训练不等于新盲测；单seed不作显著性声明。旧GeoPRR历史像素和完整权重链尚未重新认证。旧模型和全部负结果保留，不延长本预算、不用目标补训，RGB／权重／特征／原几何和private trace不公开。
