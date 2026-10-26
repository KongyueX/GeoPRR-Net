# Conditional source-view gradient control

前瞻协议已冻结，新3代码已完成，7项必要NEWCPU检查PASS（0realMNV4/RGB/cachedweights/GPU）；当前尚未启动实际队列。尚未执行metadata准入、真实模型构造、训练或GPU，也没有native/CF/真实外域结果。不是正式V32或推广模型。

问题：在已有source scalar y条件下，h1280能否识别original/roll生产程序标签；若可学，反转仅该分支对encoder的梯度是否改善source布局泛化并保护准确图。Z不是camera角度真值或错误因果。原RGB/CNN-GAP-point推理路径保留；既有metadata标签只作训练辅助监督，不能称完全scalar-loss-only。

先metadata-only验证全部5496行的view/parent-slot和未改production target值保持；再唯一source0opt见证；fresh15E detached-critic control须通过原constant读数学习gate及末轮在线critic accuracy≥.65，才评分其人工CF并fresh同15E reversal arm。唯一encoder critic梯度0→-.05，critic自己两arm都优化相同BCE。Reversal须同时保护identity15/control两view的mean/tail/max/>5/各自准确集并通过预定有限critic effect，不能挑checkpoint/seed或调λ救过。

完整定义见[protocol.json](protocol.json)与[中文协议](../../GEOPRR_V2_CONDITIONAL_VIEW_ADVERSARY_SOURCE_CONTROL_CN.md)。配置和必要NEWCPU见config.json/cpu_checks.json；实际执行尚未开始，本目录没有模型成绩。

## Limitations

训练critic准确率是在线共训统计，不是独立held-out probe，不能认证条件互信息0、因果、显著性、新盲测或真实域泛化。只用已查看source开发集和人工CF；原标签派生及历史pixel/full-weight资格不在此重认证。Source03与真实Industrial/RF不参与监督、模型调用、适配或训练选择。旧q监督与Sobel等关闭方法保持关闭。
