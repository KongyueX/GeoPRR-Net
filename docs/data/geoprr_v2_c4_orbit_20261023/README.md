# v19固定C4轨道语义池化

独立15轮轨道模型协议已冻结，5项新CPU检查与真实源RGB零更新见证通过，正式源训练于北京时间2026-10-23 16:42:41启动，没有新完整性能结果。

每张256 ROI固定生成0/90/180/270度四个张量旋转，同一完整ImageNet MobileNetV4一次编码4B视图，在1280维semantic上取FP32均值后执行同一个point和sigmoid。训练、验证、外域均使用这一个函数。原图物理scalar不变，无插值/裁剪/反射/GT几何，不选择角度或权重。模型8435793参数/464 state，新增参数0。

15轨道epoch固定训练CNN视图1115400，等于v16训练视图预算；原图scalar呈现278850、17430次反传尝试，验证22590原图行/90360视图/1425调用。两套外域1546原图/6184视图/98调用；正式总1211944视图/18953encoder调用。每原图推理四视图，推理成本增加必须记录。源15次光度抽样/选择机会与16原图×4相关方向BN batch不同于v16，不声称优化完全匹配。

实际见证使用4张unique源0/1 RGB：新模型464初始state与同seed fresh direct参照相同；训练一次4原图→16 CNN视图、finite fullencoder/neck/point梯度，0optimizer；eval两次分别原图/90度输入，观测到scalar及pooledsemantic差均0。理论C4不变性仍以浮点容差声明，不能把4例观测扩成逐位保证。共3次CNN/48视图/12原图呈现/1次反传/0optimizer，77BNcounter源+1/eval额外0，464state严格reload。正式训练重新初始化，不使用见证BN。

只用18590源训练/1506源验证，全部15轮后才各做一次source-selected best冻结外评。源RGB基础子合同与v16相同，整个模型输入合同已改变为固定C4轨道；source和external同策略，不做目标训练、BN适配、总体统计拟合或checkpoint/角度策略选择。

- [启动记录](launch.json)
- [冻结协议](../../GEOPRR_V2_C4_ORBIT_ZERO_SHOT_PROTOCOL_CN.md)
- [实际源运行检查](runtime_preflight.json)
- [新CPU检查](cpu_checks.json)

## Limitations

C4约束只处理90度整数倍整图roll，不解决所有透视、刻度方向或连续旋转；池化可能丢失有用表示，当前没有泛化收益证据。CNN视图预算相同不等于相同独立光度抽样、BN统计、选择机会或计算时间，目标推理有四倍视图成本。两外域已查看，0目标训练不等于新盲测；单seed不作显著性声明。全部旧模型与负结果保留。
