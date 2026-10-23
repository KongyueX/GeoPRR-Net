# v18完整语义与源结构辅助监督

新独立源域表示实验的协议、配置和60轮预算已冻结；8项CPU检查、18590张源标签的新解析合同检查和实际源RGB零更新检查均通过，正式训练待启动。这里没有新训练性能。

使用fresh完整ImageNet MobileNetV4和直接scalar读出，同一encoder增加训练期FPN／参考角色辅助分支，以源解析进度U、ring／arc mask M及参考角色监督共享表示。推理只执行完整semantic→point，不运行辅助decoder，也不使用目标GT几何。总训练参数8693701／498 state，活动推理通路8435793／464，辅助257908参数／34 state；源loss固定SmoothL1(beta.05)+0.1*(roleNLL/(3log256)+fieldL1)，不按外域调权重。

全部18590源行解析U/M和角色标签通过新合同检查；同v16 RGB人口、路径、物理target及bbox子合同精确一致，新增几何子合同单独记录。M最小193网格点，原物理校准限制未放宽，角色two-hot质量误差0。该合同使用U与M，不生成Gaussian Q或Q×M×U组合GT读出，也不声称旧Q支持／读数审计已在64网格重新通过。

实际4张unique源0/1 RGB见证：新模型原encoder／point464 state初始化与同seed v16 fresh ImageNet参照逐项相同；单次训练CNN前向同时产生1280semantic和四尺度特征，CNN／neck／point／全部辅助模块梯度有限且有非零信号，0optimizer。第二次eval只执行direct，辅助调用0；77个BN counter源训练+1，eval额外0。合计2CNN／8图／1次combined反传／0optimizer，498 state严格reload。正式训练会重新初始化，不使用见证后的BN状态。

预算为60完整源epoch、18590unique／1506源val、256 RGB、seed20262020、train64／val64／workers4；全部60结束后只对source-selected earliest best各做一次Industrial1395／RF100151外评。两域0训练／总体统计拟合／BN适配／checkpoint和超参数选择。新增auxiliary计算成本单独记录，不将相同CNN调用称为同计算量。

- [冻结协议](../../GEOPRR_V2_SEMANTIC_AUX_ZERO_SHOT_PROTOCOL_CN.md)
- [全源解析合同检查](analytic_source_contract.json)
- [实际源运行检查](runtime_preflight.json)
- [新CPU检查](cpu_checks.json)

## Limitations

本轮检验训练期结构监督的表示学习，部署并不新增几何读出。固定辅助权重可能负迁移，当前检查仅说明实现和输入成立，不证明泛化收益。Industrial／RF100是已查看开发集，0目标训练不等于新盲测；RF100 scalar为四点离线派生。旧模型和全部负结果保留，RGB／权重／原几何／特征及私有数据合同不公开。
