# 完整语义表示与源域角色辅助监督

2026-10-23。独立v18协议与预算冻结，8项新CPU检查与全18590源解析合同检查通过，真实RGB零更新检查通过，正式源训练于北京时间2026-10-23 15:07:10启动；没有新完整训练成绩。用户授权继续zero-shot研究，Industrial1395和RF100151不得用于训练。本轮从已有ImageNet缓存重新初始化，不恢复或延长已完成v16／v17。

## 问题与单次设计

v16完整语义直接读数的Industrial／RF100为11.686054／10.164776%FS；v17冻结该表示后加入小空间残差，结果11.690235／10.182052%FS，没有外域增量。v17预测改变量均值仅约0.07–0.08%FS，原严重错误保留。这只约束该冻结小头配方，不能据此认定空间信息无用。

本轮检验：在保留完整语义直接读数的同时，通过真实源几何与参考角色监督学习共享CNN表示，能否提高zero-shot效果。新增训练辅助结构，部署时仍用直接scalar输出，不能将它描述成新增推理几何读出或已实现角色融合。

[Hypercolumns](https://openaccess.thecvf.com/content_cvpr_2015/html/Hariharan_Hypercolumns_for_Object_2015_CVPR_paper.html)提供组合多尺度语义／定位特征的一般动机；[Kendall等的几何与语义多任务研究](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)说明共享表示及任务权重值得检验。这里不实现其uncertainty weighting，也不由这些论文推断本任务必然有效。

## 模型与监督

先按v16构造完整`mobilenetv4_conv_medium.e500_r224_in1k`和Linear1280→1，保留conv_head／norm_head及同seed下的base／point初始化，再初始化辅助模块。只用已有ImageNet缓存，HF_HUB_OFFLINE=1；不加载v5／v6／v16／v17任务权重，不下载新foundation。

同一次encoder前向取得完整1280维semantic，以及stride4／8／16／32的blocks.0／1／2／4特征，实际通道48／80／160／960。新增64通道FPN和local ReferenceRelationStateHead，沿用现有径向采样、预测center／SPD与参考条件角色逻辑；不新增第二CNN。全CNN、原point和辅助分支均参与源训练，BN只在源训练更新。

训练输出包括直接标量及辅助角色／校准；验证和外评只计算`sigmoid(point(semantic).float())`，跳过辅助decoder。所有forward接口仅接收RGB，GT H、field、mask、span及角色标签只进入源loss。P是半径1的虚拟dial-plane状态点，不是实体针尖。

唯一总loss预先固定为：

`SmoothL1(direct,y,beta=0.05) + 0.1 * (role_NLL/(3*log(256)) + field_L1)`。

角色NLL和field L1复用已有参考条件定义，去掉reading-CDF；不构造最终角差／跨度读出，不加入融合MLP、router或硬confidence门控。系数0.1是本次固定选择，不根据两套外域调节或搜索。预期训练总参数8693701／498 state，其中新增辅助257908参数／34 state；部署活动通路仍为8435793参数／464 state，实际构造核实后记录。

## 新的解析U／M与角色数据合同

沿用18590张源训练图及原1506验证。256 RGB、LINEAR resize、ImageNet归一化和原光度策略与v16一致，无配对重复曝光、新渲染、ROI扰动或新背景。训练新增解析监督，不能声称完整data contract与v16标量合同相同；应分别记录相同RGB子合同和新增几何子合同。

新训练数据入口通过原`GeoPRRV2Dataset(require_geometry=False)`读取同一RGB。已盘点全部18590行具有原图1920×1080尺寸及所需物理几何metadata、无roi_path；原入口仍检查实际解码尺寸。根据该实际数据声明与原bbox，用相同floor／ceil／边界clip／half-pixel公式取得原图→256ROI矩阵，再调用原dial_calibration。其有向C4／ordered ticks、物理y、radial0.02、center0.05、anchor0.01等校准限制保持不变。

在64×64网格上解析计算进度field **U** 和ring／arc mask **M**，以及源H／span和A／P／B角色相位；使用原field定义，要求mask非空、数值有限，虚拟P仍须在ROI内，不能裁剪或改标签。**本轮确实使用U和M，不生成Gaussian点后验Q，不计算Q×M×U组合GT读出。**

原完整geometry_targets及其point_sigma、min_support_mass0.15、max_readout_error0.01全部保留且不修改。由于新loss不用Q，该解析合同没有Gaussian Q的support／readout审计，不能把旧192网格的组合GT完整性结论转写为新64网格已通过。也不能通过修改旧dataset属性绕过output_size=image_size/4约束；新入口正常使用256／64，仅生成实际消费的解析监督。

正式训练前，必须对全部18590源行做一次新解析合同CPU检查：同v16 RGB人口／物理target／路径对齐、原物理校准限制、U／M有限且有ring支持、虚拟P在ROI内、实际FP32 H／y／span的reference targets有限、SPD正定／center在内／角色two-hot质量为1。无CNN和RGB解码，不重复旧Gaussian GT／教师检查。首个失败即保存partial并停止，不删行、重采样或放宽限制。验证与外域沿用纯scalar RGB入口，不生成辅助GT。

## 固定训练和选择

独立60完整epoch，seed20262020，train64／val64／workers4，AMP、fresh AdamW、weight_decay1e-4、gradient_clip5。LR与v16相同：epoch1–30为`3e-4*(1+cos(pi*(e-1)/30))/2`；31–60为`3e-5*(1+cos(pi*(e-31)/30))/2`。不扩seed或officialtest。

只按完整1506源验证的direct NMAE选择最早最优best，last60仅保存源结果。所有60轮完成后，才对冻结best各做一次Industrial1395／52组、RF100151／35组外评。两域不进入梯度、loss、总体统计、BN、早停、checkpoint或超参数选择；RF100 scalar仍由四点离线派生。推理仅原生ROI RGB，不使用目标GT几何、阈值／温度／TTA调整，完整失败分母error1。

每轮18590训练图／291调用、1506验证／24调用。正式计划训练1115400图／17460反传尝试、验证90360图／1440调用，外域1546图／25调用，总1207306图／18925调用。实际有效update和AMP skip之和须为17460。辅助decoder的计算、时间和峰值内存另记录，相同CNN调用数不表示相同计算成本。运行检查成本单列，不报计划为已完成。

复用已完成v16作为同源人口、输入和预算参照，不重训它。先构造相同direct模块，检查完整464 state初始化与同seed参考相同，再增加辅助分支；不声称额外模块后的所有随机执行逐步相同。该试验共同改变训练结构与监督，不作单一neck、损失权重或新理论的因果认证。

## 执行与完成

入口`experiments.run_geoprr_v2_semantic_aux`，独立输出`runs/geoprr_generalization/iteration_v18_semantic_aux_20261023/source_only_training`。启动要求：新全源解析合同通过、本轮针对性CPU检查及真实源RGB运行见证通过、代码版本冻结、旧v17 complete／父子退出、实际GPU空闲。始终单队列串行source60→Industrial→RF100。

新的真实源运行见证使用实际0／1端点RGB，检查256输入identity、完整ImageNet初始化与semantic neck、单次CNN的多尺度和direct／aux形状、有限loss和CNN／neck／point／辅助梯度、0optimizer、源BN更新及eval不执行辅助结构、完整state严格reload。不得用见证更新后的状态初始化正式训练。

保存完整model／optimizer／scaler／RNG／history／source best与last逐图、真实累计成本和两个数据子合同。真实故障先查当前step日志与活命令，保留失败／partial，必要实现修复后在同output从latest恢复；不删样本、绕保护或静默改配置。

全阶段结束后仅一次必要独立CPU数值与执行核对，再比较source／外域NMAE、Acc、zero／nearzero／nonzero、组、ordinary1495／1497、severe6、3172／2810／3663／9323／8195、tail／max／救回新增和实际成本。只公开显式allowlist的规范化结果和代码；RGB、权重、H／C／XYZ、bbox、特征、raw run／data_contract和private trace保留私有。

实际执行代码46a0637，父414650／初始训练子414655；父子精确命令、唯一GPU及首批64源图有限梯度已核实。首批primary0.234319686890、normalized role0.999382257462、field0.007212311029、total0.334979146719。见[启动记录](data/geoprr_v2_semantic_aux_20261023/launch.json)及[实际源输入见证](data/geoprr_v2_semantic_aux_20261023/runtime_preflight.json)。这些是执行证据，不是训练性能。

## Limitations

辅助任务可能与direct目标冲突，固定0.1不保证有利；source改善不保证外域改善。训练辅助结构不等于部署时增加局部几何能力。两外域是已查看开发人口，0目标训练不等于全新盲测；单seed不作显著性或普适泛化声明。旧GeoPRR历史执行像素与完整权重链尚未重新认证。全部旧结果、模型和失败保留；无增量不机械延长60轮或继续叠加模块。
