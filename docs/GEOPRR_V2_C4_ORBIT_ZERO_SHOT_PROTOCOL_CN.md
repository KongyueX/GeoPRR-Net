# C4轨道语义池化：固定源域视图预算实验

2026-10-23。独立v19结构、预算和选择规则冻结，代码与新运行检查准备中，尚无新训练结果。Industrial1395和RF100151不用于训练或适配。旧v16／v17／v18已完成，全部模型和结果保留，不恢复或延长旧任务。

## 问题与模型

v18的源结构辅助监督改善了部分参考型错误，但存在混合迁移：相对v16，RF100均值降低2.3121%，Industrial升高12.9081%，RF100严重尾部也回退。下一步单独检验一个任务对称性：整张表盘ROI旋转90度的整数倍，物理读数和针线相对刻度的关系不变。

本模型在训练和推理中固定执行同一函数。令R_k为torch.rot90的k次逆时针90度旋转，k∈{0,1,2,3}：

`pooled(x) = mean_k encoder(R_k(x))`，`reading = sigmoid(point(pooled(x)))`。

四个旋转发生在256×256、ImageNet归一化后的RGB张量上，是无插值、无裁剪的像素重排；不加入反射、任意角度、缩放、ROI或角度策略搜索。整个ROI一起旋转，scalar y不变；不改源文件、读数、H或其他原标注。模型和loss均不使用这些几何标注。

同一组共享权重将k-major排列的4B图像一次送入完整MobileNetV4 encoder，取得4B×1280 embedding，转换FP32后reshape为4×B×1280并取均值，再执行Linear1280→1和sigmoid。池化对象是semantic／logit，不能替换成四个sigmoid概率的平均。只旋转CNN后的特征图再做GAP不会获得该模型，必须在encoder之前旋转RGB。

eval时，C4旋转输入只会排列这四个embedding，所以理想实数运算下模型对C4不变。实际浮点归约及batch kernel可能产生小差异，运行检查使用明确容差，不声称逐位相同，也不保证连续旋转、反射或尺度方向混淆已解决。

完整encoder保留conv_head／norm_head和1280维语义，预计8435793参数／464 state，与v16 direct参数键相同，无新增可训练参数。仅使用已有ImageNet缓存fresh初始化；不加载旧任务权重、不下载新foundation。全部CNN和point参与源训练，BN只从源视图更新，eval冻结。

[TI-Pooling](https://openaccess.thecvf.com/content_cvpr_2016/html/Laptev_TI-Pooling_Transformation-Invariant_Pooling_CVPR_2016_paper.html)提供在共享分支输出上做变换池化的结构动机；[Group Equivariant CNNs](https://proceedings.mlr.press/v48/cohenc16.html)提供利用任务对称性的背景。此处不是其group-convolution实现，也不复制TI-Pooling的最大池化；采用固定C4均值的共享encoder模型，不作新理论或本任务有效性认证。

## 源数据和优化预算

沿用18590源unique与原1506验证、256 RGB／LINEAR／ImageNet和原光度增强，无新渲染。每次原图先采样一次光度增强，再由该张量产生完整C4轨道；同一轨道四视图共享光度。只用scalar SmoothL1 beta0.05，不使用v18辅助decoder、角色loss、teacher、router或GT几何。

固定15完整轨道epoch，原图batch16、encoder batch最多64、workers4、seed20262020，fresh AdamW、weight_decay1e-4、AMP、gradient_clip5。每个loss针对池化后的B个标量，而非把4B个视图当作4B个独立标签。

训练计划：每轮18590原图呈现、74360个CNN视图、1162次encoder调用和反传；15轮共278850原图呈现、1115400个CNN视图、17430次尝试更新。最后原图batch14，对应56视图，reshape必须使用实际B。

学习率沿用v16两段60虚拟epoch的函数，按实际尝试更新序号s∈[0,17429]推进：

`virtual_epoch = 1 + floor(s * 60 / 17430)`，`lr = v16_learning_rate(virtual_epoch)`。

虚拟1–30使用3e-4余弦，31–60使用3e-5余弦，公式与v16原函数相同。AMP skip仍计入s；不能按有效update推进。第8715个零基尝试进入虚拟31，处于轨道epoch8内部，因此每轮必须记录实际LR区间／虚拟epoch计数，不能只填一个不真实的epoch LR。有效update＋AMP skip须为17430。

该方案匹配v16的训练CNN图像预算1115400，但不是完全相同优化实验：每原图只有15次光度抽样／scalar loss，而v16为60次；源BN batch为16个独立ROI×4相关方向，而v16为64个独立ROI；学习率每虚拟epoch290或291次尝试，而v16每轮291次。参数相同和CNN视图数相同不等于相同计算／时间或逐步随机实现。

## 源选择、外评与分母

每轮用同一个固定C4模型完整验证1506原图，只按源NMAE选择最早最小best，共15次选择机会；v16有60次，差异照实记录。last15仅保存源结果。不增加额外源验证次数来搜索checkpoint。

15轮验证为22590原图行呈现、90360个CNN视图、1425次encoder调用。验证最后原图batch2对应8视图。只有完整15轮结束后，才对冻结source-best各做一次Industrial1395／52组、RF100151／35组评估。

Industrial为5580 CNN视图／88调用，RF100为604／10调用；外评合计6184视图／98调用，是单视图模型的4倍CNN图像曝光，不能称免费或同推理成本。每原图仍只输出一个标量、保留一个error分母。输入失败的原图产生四个placeholder视图，但只记一次normalized error1，明确分别统计failed originals和placeholder CNN views。RF100 scalar由四点离线派生，量程及target只用于计分。

正式全阶段计划1211944 CNN视图／18953 encoder调用；原图呈现、视图、调用、反传、有效update／skip、时间和峰值内存分别实记，不报计划为已完成。模型训练、验证、目标均使用同一预先固定轨道函数，没有目标域选择角度、融合权重、BN、阈值、温度或其他策略。

RGB基础子合同与v16相同，但整个模型输入／推理合同已变为固定四视图，不能写成与v16单视图输入完全identity。原14494／1506／4000和ordinary1495／1497／severe6及历史重点名单保持。

## 实现与检查

独立入口`experiments.run_geoprr_v2_c4_orbit`，输出`runs/geoprr_generalization/iteration_v19_c4_orbit_20261023/source_only_training`。不修改旧v16／v18训练与eval函数；新orbit evaluator用原图batch16，实际encoder计数为4N视图，scalar行数仍为N。

启动前完成针对性CPU检查：k-major分组／FP32 embedding均值、非旋转不变toy encoder的C4池化、最后partial batch、失败原图与四视图分母、虚拟LR边界及恢复尝试计数。真实源RGB运行见证只用已有0／1源图，确认完整464 state初始与同seed fresh direct一致、四视图像素重排、1280维语义池化、有限encoder／neck／point梯度、0optimizer、源BN更新、eval C4不变性及完整state严格reload；不使用目标RGB或重跑旧几何GT。

正式运行必须v18全部complete且父子退出、GPU真空闲、代码／协议／预算冻结，新源见证通过。单队列依次source15→Industrial→RF100。保存完整model／optimizer／scaler／RNG／history／LR尝试状态／累计计数、源best与last逐图及真实成本；故障保留partial，在同output从latest按实际记录恢复，不删样本或改协议绕保护。

完整阶段后仅一次独立CPU数值与执行核对，报告各域NMAE／Acc、zero／nearzero／nonzero、组、ordinary1495／1497／severe6、重点3172／2810／3663／9323／8195与sync9／RF41、tail／max／救回新增及每原图4视图的成本。只公开显式allowlist的规范化标量结果与本任务代码，RGB／权重／原几何／特征和private trace保留私有。

## Limitations

C4池化可能抹掉有用特征或平均不兼容表示，不能预先承诺泛化收益。其保证仅为固定BN eval下的C4模型对称性，不是所有旋转／透视／量程或刻度方向不变性。训练原图呈现、光度独立抽样、BN相关性、source选择次数和LR粒度均不同于v16，CNN视图预算相同不能证明纯单因素因果。两外域已查看，0目标训练不是新盲测；单seed不作显著性或普适声明。旧GeoPRR历史像素／完整权重链尚未重新认证。无收益保留负结果，不延长15轮、追加seed或扫目标角度策略。
