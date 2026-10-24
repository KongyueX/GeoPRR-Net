# RGB512完整语义直接读数：zero-shot协议

2026-10-24。V26检验单张原生ROI的输入与encoder空间采样表示：RGB512经fresh完整缓存ImageNet MobileNetV4，完整semantic1280接新Linear1280→1／sigmoid。全部CNN与point源训练，唯一scalar SmoothL1。源人口、光度、R及停止规则保持；不新增RGB、延长旧训练或堆模块。实际8435793参数／233个参数tensor／464 state；真实512源AMP见证、完整源109轮与best89两域各一次冻结外评均已完成，唯一保存数字／执行核对73307项PASS、0失败。结果见[完整报告](data/geoprr_v2_high_resolution_direct_20261024/completed/report_cn.md)，不升级默认。

## 源证据与表示问题

V25源best37原1506 mean／tail10／max为0.427194／2.031386／58.511108%FS，roll512为0.785448／3.387103／17.675227%FS。源低均值仍伴随严重错误。V24固定主干的深层adapter源验证精度较弱；下一方向保留完整可训练语义，改变输入和encoder内部空间采样，不继续扩展adapter或依赖旧task输出。

只读取指定8张源图的文件header和既有ROI尺寸，未解码像素、运行模型或重审GT。rolltrain row0／3 native ROI各919×919，row4／7各593×593；原val row0为507×507、row1为541×499；rollval row0／1各469×451。八例两边均>256，支持检验512相对256减少下采样、改变空间采样的可执行问题。部分原ROI小于512，不能声称都恢复原生细节，也不能以8例代替源全体统计或证明严重错的成因。

旧GeoPRR／V12使用768及其训练历史不能作为分辨率单因素证据。V26与V25的输出表示及loss也不同；与V22的源风险选择／历史不同，不宣称纯分辨率因果。方向由源header、源严重错和静态接口选择，不使用目标分数或病例选择512。

## 单一模型与新输入合同

完整`mobilenetv4_conv_medium.e500_r224_in1k`，`num_classes=0`，仅既有缓存ImageNet权重，`HF_HUB_OFFLINE=1`。保留8434512参数／462 encoder state、conv_head／norm_head与实际semantic1280；实际reported `num_features=960`不替代输出维度。新point weight Normal(std=.001)、bias0，logit转float后sigmoid，唯一B标量。完整原网络与point共同源训练，BN只在源训练更新、eval固定；不加载旧task checkpoint。

`native ROI → LINEAR resize512 → ImageNet normalize → 一个完整encoder → semantic1280 → 新point → reading`。

静态安装版本使用adaptive global average pooling到1×1，支持空间尺寸改变而保留完整neck；本轮保存见证已确认实际512输入、1280语义及必要梯度。没有CDF／distribution、adapter、FPN、query、teacher、GT角色／几何读出、router、输出融合或TTA。模型只接受image。实际CPU零前向构造确认8435793参数／233个参数tensor／464 state，全部参数可训练。

新source dataset必须显式`image_size=512, output_size=128, require_geometry=False, pair_manifest=None`。128满足数据类的image_size/4接口，不生成或消费几何场，也不认证128网格的旧Gaussian Q／support。原V16 ScalarRGBDataset及其failure占位硬编码256，不可直接用于V26。新source失败占位必须3×512×512，推理和loss仍只RGB／scalar。

四份manifest不变：旧18590训练、原1506验证、既有4096roll训练、512roll验证。可只调用V22 source_populations作原有metadata／roster／split／layout／seed／scene guard，再用相同manifest重建四个512 dataset，逐项rows直接等于原四dataset；不得为核实roster额外getitem、旧RGB前向或GT检查。

新data contract将旧populations保留一次，其余旧合同仅作`population_provenance`，单列权威`RGB_input_contract`为512／128、LINEAR／ImageNet／原native ROI／原光度／geometryFalse。旧source历史中若有256说明不改写成新部署；RGB子合同和整体部署合同均已改变，不能说输入等于V16／V25的256流程。严格resume使用本轮新合同和配置。

ExternalPopulationDataset使用本轮512／128配置，仅对给定原生ROI作相同处理，不新增目标GT裁剪、中心、针尖、H或统计适配。

## 数据、源loss与停止

源22686unique完全复用：old18590一次＋roll4096四次，每epoch34974呈现／547批64末30；验证原1506／24调用＋roll512／8调用，共2018／32。原contrast／saturation .8–1.2、brightness ±.1保持。没有新增RGB、增强、名单、标注、source seed或物理渲染；原14494／1506／4000与fixed ordinary1495／1497／severe6及focus保留。

原1506 scene guard不变；roll512独立layout／ID／group／声明generated seed但sharedHDR，原20096 metadata缺seed，不认证全部历史seed链或新域盲测。scalar目标由原manifest保留，不因512重生成几何或重新标定。

唯一loss恢复为scalar SmoothL1(beta=.05)。fresh AdamW LR3e-4／wd1e-4／clip5／default fresh GradScaler／AMP／seed20262020／train64／val64／workers4。encoder／point AMP F16、最终scalar FP32，固定精度不扫描。训练预算按该新表示需要执行，不以相同CNN次数声称同成本。

源R仍为`(mean_original+mean_roll+tail10_original+tail10_roll)/4`，normalized full errors／失败error1，tail严格ceil10%为151／52。最早严格最小R选best，同R驱动Plateau factor.3／patience8／threshold0abs／cooldown0／minlr1e-6／eps1e-8和停止。min60／max160，epoch≥60且epoch−best_epoch≥20停止，counter从1计，LR／resume不重置，不以目标延长或扫描384／512／768。

完整model／optimizer／scheduler／scaler／RNG／history／totals／init及新source合同保存，严格本轮protocol／config／512输入恢复。训练统计和部署均为同一scalar通路，last只源。

## 必要新检查

只检查512输入重建、同roster和新合同，不重跑旧R数学、GT、教师、G0、旧tests或旧模型。CPU toy验证512 dataset真实LINEAR／normalize／无几何、失败占位512、四人口rows与曝光不变、拒256恢复；toy完整语义单CNN／point、RGB-only拒GT、B1/B3 eval partial、必要梯度和strict state。真实model CPU只零forward构造，0cached权重／RGB／GPU，实际metadata由构造记录。

一次真实源AMP零更新见证：4张已有roll训练0／1图＋两源验证各2张，3CNN／8图／1combined backward／0optimizer。记录实际B×3×512×512输入、完整semantic1280与point1280→1、scalar与F16／FP32通路、CNN和point实际调用3／8、encoder／stem／完整neck／point有限非零梯度、初始完整cached462逐项保持、233参数无optimizer不变、77BN源+1／eval额外0、完整464 strict reload、两验证对原nativeROI的512 LINEAR／ImageNet identity。256实际输入不能通过该见证。

pref反传固定loss×128后grad÷128，formal default fresh GradScaler不改、不扫描scale；正式fresh重建不沿用prefBN。见证仅执行／数值检查，不是准确率。V25整个完成并经ONE核对归档、实际父子退出／GPU空闲、V26代码／配置冻结后才串行pref→source→Industrial→RF100。活job只监控queue／currentstep原日志／history／latest与真实父子GPU，不改livecode／model／config／名单／标签、不增加模型评估，4workers不是额外模型。

## 冻结外评、实际成本与归档

源完整停止锁定best后，Industrial1395／52与RF100151／35各一次冻结评估，last不外评。两域为已查看开发人口，不训练、总体统计拟合、BN适配、checkpoint／epoch／早停／LR／训练超参／分辨率／ROI／阈值／温度选择，不用目标病例改标签或采样。forward只nativeROI RGB/eval，RFscalar离线四点派生，输入／scalar失败完整原图分母error1，不删行或把有效饱和读数误叫invalid。

E轮实记train34974E图／547E CNN及backward尝试，val2018E／32E调用，updates＋AMPskip=547E；外域1546／25／0back／0update，正式总36992E＋1546图／579E＋25调用。point实际hooks每次与CNN一致。512单图输入元素是256的四倍，激活、compute、数据处理及time／peak由实际记录，不称相同成本或四倍实际wall；pref单列，旧源生成只引用不重算新费用。

完整后ONE新鲜独立CPU保存数字／执行核对，收齐两源best／last1506＋512及完整I/R逐图／groups／summary、source选择与外域时间隔离、actualcounts／time／peak／init／stop／resume，显式规范化allowlist推现有main。报告mean／Acc1/2/5／median/p90/p95／tail10／max、相对v16／旧GeoPRR／v24／v25及保存参照的>1/2/5新增救回、group／zero／nonzero／nearzero、fixed cohorts／focus。roll512只自身报告，不补评旧模型。原RGB／weights／H/C/XYZ/U/M／bbox／features／rawrun／fullcontracts／sourcearrays／privatepaths保持私有。

## 完成记录

源109轮完整结束，earliest strict source R选best89，R=0.010823831190581175（1.082383119058%FS）；109−89=20，按min60／max160与patience20停止，无resume。LR在49／82／98／107轮后下降未重置counter。两域只在源完成后各一次冻结同一best89，last109只源；四阶段complete／exit0、父进程退出，收集时GPU空闲。

源best原1506 mean／tail／max为0.329670／1.376887／17.664927%FS，roll512为0.504289／2.118687／11.576979%FS。Industrial mean22.861047／tail64.456899%FS弱于v16、旧GeoPRR与V25，较V24改善；相对V25的>5总数833对917（救回191／新增107）。RF100 mean9.059529%FS、Acc@1／2／5为33.774834／54.304636／71.523179%，相对v16 >5总数43对58（救回30／新增15），但tail51.670712／max70.405841%FS升高，mean仍弱于旧GeoPRR与V24。正负结果均保留，不升级默认。

正式CNN／point各4033674图／63136调用，源backward尝试59623＝59600更新＋23AMPskip；pref3CNN／8图／1back／0optimizer单列。实际512训练pipeline18503.923992秒、source full19355.291917秒，source peak allocated／reserved8262538752／11347689472bytes；一次CNN不等于同成本。完整cached462仅初始化保留，全部CNN／point随后源训练、BN源更新／eval冻结；权威RGB512／128改变输入合同，旧256／64仅历史人口provenance。

唯一新鲜独立CPU保存数字／执行核对73307项通过、0失败、最大差1.4210854715202004e-14，不加载模型／weights／RGB／checkpoint，不重跑GT、旧测试或外评。[completed报告](data/geoprr_v2_high_resolution_direct_20261024/completed/report_cn.md)、[指标](data/geoprr_v2_high_resolution_direct_20261024/completed/metrics.json)、[比较表](data/geoprr_v2_high_resolution_direct_20261024/completed/comparison.csv)与[26项completed allowlist](data/geoprr_v2_high_resolution_direct_20261024/completed/public_allowlist.json)已规范化归档；[总allowlist](data/geoprr_v2_high_resolution_direct_20261024/public_allowlist.json)33项。fullcontracts、rawrun、源人口／fixed-cohort完整ID数组、完整delta／threshold／rescue／new名单、RGB、weights、geometry、features、bbox、原BN值与privatepaths保持私有。

## Limitations

八例header只证实可用源ROI尺寸，不证明全部source细节、针线可辨性或错误成因；部分ROI小于512不能恢复原来没有的信息。高分辨率、回到scalar读出／loss及训练历史同时变化，不作单一模块或分辨率因果声明。保留完整语义不保证整体泛化，源精度不能替代外域mean／tail／max和新增大错。单seed、两个已查看开发域、共享HDR源验证、旧GeoPRR历史pixels／完整权重链未重新认证的边界保持，不报SD、显著性、新盲测、普适性或新理论。所有旧模型、失败、G1／anchor／65轮与固定人口保留，不自动升级模型。真人授权单轮归档后持续独立源域结构研究，不延长已完成配方或重复外评。
