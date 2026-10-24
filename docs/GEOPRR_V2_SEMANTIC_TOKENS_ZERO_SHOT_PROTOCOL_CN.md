# 全语义与多尺度空间latent tokens：zero-shot协议

2026-10-24。独立v23按以下冻结结构与source mean-tail R规则完整完成117轮和两套冻结外评，最早source-best97；两域mean、最差10%及max未超过v16或旧GeoPRR历史参照，不自动升模型或延长。唯一保存工件CPU审计73780项通过、0失败，完整结果见[完成报告](data/geoprr_v2_semantic_tokens_20261024/completed/report_cn.md)。实际模型9374417参数／293个parameter tensor／528 state，执行版本90d932cc850f4e963c89e23b766daf19fe6f5ca4。本文保留执行前冻结的方法与选择规则；两外域仅在全部source结束后评估锁定best，全部旧模型和失败保留。

## 可证伪问题

[v22完整结果](data/geoprr_v2_roll_source_20261024/completed/report_cn.md)显示，两域Acc@1／2／5提高，但均值和最差10%均值弱于v16，不能用主体阈值准确率替代尾部目标。本轮检验完整分类语义与保留细节的多尺度空间信息能否共同参与可学习关系读出，从而减少严重错误并改善整体泛化。判定时同时报告mean、tail、max及错误交换，不能凭一个case一票否决，也不能仅用Acc提升宣称达成目标。

[DETR](https://arxiv.org/html/2005.12872v3)中learned queries通过attention读取图像空间关系，[Perceiver IO](https://arxiv.org/html/2107.14795v3)提供用query接口从latent表示形成灵活输出的一般动机。本轮不复制检测box／Hungarian set loss或完整Perceiver latent encoder，仅使用两层标准decoder汇总图像空间memory。这些论文不证明表计zero-shot或严重尾部收益，attention本身也不保证细针问题自动解决。

## 单CNN的新部署通路

fresh已有缓存的完整`mobilenetv4_conv_medium.e500_r224_in1k`，`num_classes=0`，保留conv_head／norm_head和实际1280维semantic。一次encoder前向同时取得blocks.0／1／2／4，实际通道48／80／160／960，stride4／8／16／32；不分配或保留旧v16 point，不加载任何旧任务checkpoint。只用现有ImageNet缓存、HF_HUB_OFFLINE=1，不下载新foundation。

256输入对应四级空间网格64×64、32×32、16×16、8×8。stride4特征先执行`pixel_unshuffle(2)`，将每个2×2子格的有序pattern重排到channel：C48×64×64成为C192×32×32。这一步不先对局部2×2求均值；后续投影仍是可学习压缩，不能宣称完整读出无信息损失。

四级[192,80,160,960]各自投影到128维，按token做LayerNorm，并加入固定规则二维sin／cos cell位置（每轴cell center除以该级尺寸并乘2π，统一ROI坐标约定）与learned level embedding。按空间顺序展平后，memory为1024＋1024＋256＋64＝2368个128维token。位置来自ROI规则网格，不是GT center／针尖／刻度，也不使用外域几何或总体统计。

四个通用query由semantic1280→128投影加learned4×128 query组成；它们没有实体针尖、量程start／end或A／P／B认证。两层独立构造的TransformerDecoderLayer均为dim128、4 heads、FF512、dropout=.1、GELU、batch_first=True、norm_first=True，执行query self-attention与memory cross-attention。两层分别实例化，不将一层初始化后复制成相同初始层。

最终query经LayerNorm，四token按固定顺序flatten为512，接512→128、GELU、128→1，在scalar logit转float后sigmoid，形成唯一归一化reading。

`ROI RGB → 一个完整encoder → semantic条件queries＋多尺度空间memory → 两层decoder → 四token联合scalar读出`。

全部CNN、token projection、query、decoder与readout从源数据端到端学习，BN只在源训练更新，eval固定。部署也走完整新通路，不是v17冻结encoder的小残差；没有v16 scalar残差／融合、几何校准／ring／GT角色头、router、CDF、teacher、confidence门控或TTA。每图按固定权重计算query属于普通RGB条件推理，不是目标域适配。实际零前向CPU构造确认encoder8434512参数／462 state，新readout939905参数／62个parameter tensor，完整9374417参数／293个parameter tensor／528 state（含4个固定位置buffer）。实际RGB零更新见证已记录3次CNN／8图／1次反传／0更新，shape、有限梯度与AMP路径通过、完整528 state严格reload；该见证不是准确率。

## 完全复用已完成源数据

复用v22已完成的22686 unique源训练：旧18590各一次、新真实roll4096各4次，精确34974呈现／547批，batch64末批30，每次重复独立使用原光度增强。原1506验证和独立roll512验证共2018图／32调用。没有新RGB、渲染、名单、物理标签、几何、配对或增强策略改变，256 OpenCV LINEAR／ImageNet与原contrast／saturation .8–1.2、brightness ±.1保持。

新roll验证只在layout／sample ID／group／声明generated seed层面独立，共享现有HDR；原1506仍保留旧scene guard，20096原训练／验证metadata缺少seed记录，不能宣称完整历史seed链已认证。复用[源完成记录](data/geoprr_v2_roll_source_20261024/source_completed.json)及原split合同，不重跑新旧GT、源渲染或v22运行见证。roll512精确0／1人数均为0，不作端点保证。原14494／1506／4000及ordinary1495／1497、severe6和历史重点名单保持。

模型与loss只用RGB及source scalar y，唯一训练loss为SmoothL1(beta=.05)。GT H、center、SPD、角色、field、mask和针尖不进入模型或loss；既有数据物理检查不变，不把模型读取latent tokens描述成已监督实体角色。

## 预先冻结的源mean与tail风险选择

为对应大错尾部目标，每轮在两套完整源验证上计算：

`R = (mean_original + mean_roll + tail10_original + tail10_roll) / 4`。

四项均为normalized full-denominator error；mean与tail各占一半、两个源人口各占一半，权重在外评前固定，不根据目标数值调节。tail10严格复用当前summarize的最大`ceil(0.1*N)`误差均值：原1506取151行，roll512取52行；输入或scalar失败以error1同时进入mean和tail，不剔除、不人工指定单case。R不是2018行pooled均值，也不是v22的mean-only score。

同一未舍入R统一用于最早严格最小best、ReduceLROnPlateau及source patience。fresh AdamW初始LR3e-4、weight_decay1e-4、gradient_clip5、AMP、seed20262020、train64／val64／workers4，至少60、最多160完整epoch。scheduler为mode=min、factor=.3、patience8、threshold0／abs、cooldown0、min_lr1e-6、eps1e-8，每轮两套验证完成后step，下一轮使用新LR。

早停counter从epoch1开始，仅严格新R best重置，LR下降不重置。完成epoch后若`epoch>=60 && epoch-best_epoch>=20`则停止，最多160兜底；不强凑v22的114轮，不从60才开始另累计20轮。last是实际停止轮，仅保存源结果。checkpoint恢复必须使用本轮protocol／config／R定义与scheduler／best／RNG状态，不能用v22 mean-only选择或旧任务latest继续。

训练梯度仍只来自SmoothL1，没有增加tail loss、手工难例权重或目标监督。本轮同时改变结构与源风险选择，不能解释为纯结构单因素试验；采用同一可执行学习范围不等于同预算是主要目标。

## 冻结外评与成本

完整source学习按上述规则结束、best锁定后，各一次Industrial1395／52组与RF100151／35组，只评估该单模型；last不追加外评。两外域不得用于训练、总体特征统计拟合、BN适配、checkpoint／epoch／LR、停止、阈值、ROI、温度、query维数或其他训练超参数选择。推理仅原生ROI RGB、eval、同256处理；RF100 scalar由四点离线派生。保留全部声明输入，每个失败按normalized error1计分。

E轮实际预算为train34974E图／547E次CNN及反传尝试，validation2018E图／32E次CNN；目标1546图／25次CNN、0反传／更新。正式总`36992E+1546`图／`579E+25`次CNN，有效updates＋AMP skips须为547E。每图一个CNN，但token投影／memory／decoder带来额外计算与显存，必须实记，不按相同CNN计数宣称相同成本或便宜。旧源生成成本只引用既有记录，不重复计新渲染。

报告原1506与roll512的mean、tail10、R及实际选择／停止证据；目标报告NMAE、Acc@1／2／5、median／p90／p95、最差10%／max、>5%FS救回／新增、组与zero／nonzero／nearzero、ordinary1495／1497／severe6及重点3172／2810／3663／9323／8195、sync9／4025／338／4658／615和主要目标大错。连续收益、尾部和新增大错同时保留，不用一个旧门槛自动删除方法或论文，也不用更弱对照掩盖对v16／旧GeoPRR的差距。

## 执行与归档

固定训练ID为`geoprr_v2_semantic_tokens_source_only_v1`，模块`experiments.geoprr_v2_semantic_tokens`，入口`experiments.run_geoprr_v2_semantic_tokens`，独立输出`runs/geoprr_generalization/iteration_v23_semantic_tokens_20261024/source_only_training`。[配置](data/geoprr_v2_semantic_tokens_20261024/config.json)从实际训练器CONFIG表达式静态规范化；代码／配置／协议在实际运行前冻结；实际参数及state见保存的CPU／RGB运行记录。

必要CPU检查只针对新结构与新R接口：四尺度unshuffle／memory顺序和shape、固定规则位置、独立decoder层、partial batch、RGB-only、R的ceil-tail／失败分母／strict best与patience、完整checkpoint严格加载。不重跑旧GT／G0、旧模型、v22预检或完整审计，不追加评分／查新／评审循环。

一次新实际源见证计划3次CNN／8图／1次combined backward／0optimizer：四张实际roll训练0／1图及两套验证各两张，检查semantic1280、2368 memory、4×128 query、decoder／MLP和scalar路径，source loss及encoder／neck／projection／query／decoder／readout梯度有限并与计算图相连；记录source BN更新／eval额外0、256验证输入identity、全部实际state严格reload与无optimizer参数不变。见证的三次前向实际采用CUDA autocast float16，唯一反传固定将loss乘128后逐gradient除128，不构造或step优化器，用于检查新attention的AMP kernel与梯度路径；正式训练的fresh GradScaler策略不变，不扫描精度或loss scale。只从真实执行记录声明通过，不将小kernel当准确率；正式训练fresh初始化，不复用见证BN。

确认v22已complete、原父子退出、实际GPU空闲，数据保持原已完成合同，新配置／代码冻结且CPU／RGB见证通过后，启动唯一queue：新见证→source至R停止→Industrial→RF100。以pipeline／current_step、原日志、history／latest和实际GPU命令监控，best时间不变不是停滞，阶段交接短暂空闲不重复启动。活job不改code／model／config／名单／标签，也不并起评估。

保存完整model／optimizer／scheduler／scaler／RNG／history、每轮四个源风险分量、R与LR／停止计数、source best／last的1506＋512逐图、两外域逐图／组／summary及真实调用、updates／skips／时间／峰值／resume。真实实现故障先当前step日志与活命令，保留failure／partial／snapshot，在同output按latest恢复，不删样本或绕保护；实质方法／名单变化另立实验。全部完成后仅一次必要独立CPU数值／执行核对，再形成完整结论，只有完整结果再通知，不周期性发启动／中途快照。

只推本任务代码与显式规范化allowlist到已有main，不建branch；RGB、权重、H／C／XYZ／field／bbox、特征矩阵、raw run／完整源合同／人口数组与private trace保持私有。

## Limitations

结构与source mean-tail风险选择同时改变，不能单因素归因attention；新R不保证外域尾部改善，四个latent query也没有实体语义认证。共享HDR的新roll源验证不是背景或域盲测，未取得旧v16对该新人口的R基线；历史旧GeoPRR像素／完整权重链未重新认证。两外域已查看、单seed不支持SD、显著性或普适机制，常规attention读出不包装成新理论。所有旧模型／失败与固定名单保留，不能只用Acc收益替代本轮mean／tail目标，也不因一个case一票否决；未达目标不自动promotion或机械延长／扫参。
