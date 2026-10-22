# MobileNetV4跨数据集泛化迭代

2026-10-23。用户授权自主调整当前MobileNetV4改版、参考已有论文并开展多个受控尝试，目标是提高跨数据集表现。此授权开启新的泛化实验，不改变已完成v5／v6／v7／v8的配置、名单、65轮结果或原门槛失败，也不恢复配对守恒作为主贡献。

## 本轮边界

首批仍只使用源域训练监督。Industrial-1395及RF100-151用于已查看的外部开发筛选，不称新盲测，不将目标读数或几何用于训练。旧14494／1506／4000人口、ordinary1495／1497与严重6均保留；不扩seed或重跑SyncG officialtest。采用已有seed20262020和当前服务器、项目venv，GPU串行执行，不租新算力、不改变系统环境。

## A：已有模型定位

统一原生ROI入口`experiments.evaluate_geoprr_v2_generalization`严格加载各自真实协议和结构。比较初始field best61、v3 supervised best63、v5 joint best57、v6 feedback best48及v8 global best2；v7 full best64直接复用已完成的同人口原生结果，避免重复执行。保留各自原生读出，包括field自身的appearance fallback，不给某一臂添加额外阈值、校准或GT几何输入。

每个新候选分别完整计分Industrial1395／52组和RF100151／35组，ROI RGB、原768／192和ImageNet归一化、eval、checkpoint AMP、FP32 target、完整失败分母error=1。保存逐图、组、子集、NMAE／Acc／尾部／max、真实CNN调用与图像数、wall及峰值。没有模型评估时进行的训练更新。

## B：首批短训练

A阶段明确后选择一个共同的现有角度Reader锚点，并在训练前保存精确路径、epoch、原配置、初始化与新配置。首批拟比较三个配方，各4个完整epoch，fresh optimizer、同初始化／源人口／曝光／原三项loss：

1. 低学习率续训参照，backbone BN正常更新。
2. 相同配方仅固定backbone BN运行统计；affine参数可学习。
3. 配方2加早期实际backbone块的训练期MixStyle，p=0.5、alpha=0.1；无新增参数，eval严格为identity。

共享参数学习率1e-5、state head5e-5，warmup=0；共享组仍包含backbone、lateral、refine。保留每轮源验证、best-source及last，不把初始化称为原始ImageNet特征恢复。训练入口／精确配置及资源检查另行记录后才运行。BN和MixStyle仅是待证伪干预，采用已有方法不自动构成新算法创新。

三个配方执行结束后，分别比较best-source与last的外部开发表现；同epoch的两个文件复用结果，不重复评估。按各数据集连续效果、近零／普通／尾部、源域变化与真实成本判断，允许局部回退，不用原零新增错门槛否决泛化研究。没有稳定收益则保留负结果并重审，不自动延长旧65轮或堆加loss／router。后续阶段若实施，另冻结其配置与预算。

## 文献依据与输入改进

早期MixStyle参考[Zhou等，ICLR 2021](https://arxiv.org/abs/2104.02008)；只作用于流向后续backbone的早期空间特征，不混角色tokens／状态logits或最深语义特征。BN统计与域偏移相关见[Schneider等](https://arxiv.org/abs/2006.16971)，但其目标统计适配结论不等于本轮固定源统计必然有效。[LP-FT](https://arxiv.org/abs/2202.10054)的原始预训练保留与冻结已特化best64不同，不混为同一实验。

源域精确零位缺失、ROI裁剪与外观覆盖也是后续明确候选；几何改动必须同步H与结构标签，不伪造GT。当前旧RPM detector ROI合同因缺资产保持未运行；任何新ROI准备必须单独声明来源和输入条件，不套用旧1775／22记录。

## 执行与归档

每个队列持独立锁、PID、原始日志和状态，启动前source项目env并核对实际GPU；保留partial和故障证据，真实实现故障修复后同output恢复，方法／名单／配置实质变更另实验。公开仅任务代码与显式规范化标量allowlist；RGB、权重、vendor、H／C／XYZ／像素及private trace不上传。记录原G1／anchor失败，不将新外部开发成功改写为旧门槛通过。
