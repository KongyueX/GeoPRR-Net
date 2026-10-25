# 先校准读数头再共同学习：有界源控制

唯一问题：在相同430次完整CNN反传预算下，先用冻结ImageNet表示校准同一个Linear1头，是否改善随后源ID的平均与尾部学习？模型、head、pool、norm、source数据与FT优化不变；只检验同一新run的头初始化流程，不是正式V32、LP-FT论文复现或已证实的泛化方法。

Kumar等ICLR2022的[LP-FT研究](https://arxiv.org/abs/2202.10054)提出先拟合线性头再全量微调；其回归理论简化、实际OOD证据为分类，并依赖较好的预训练特征。这里不认为已证明表征被破坏，也不从该论文推出仪表收益。V15实际冻结的是source训练过的v6/960维GAP，而非freshImageNet1280；V16另起fresh并未继承V15头，不能据二者串行关系认定真正LP→FT已做。V24 source原2.18%并非完全未学，V27非有限失败也不足以判定所有固定统计策略。L2-SP会引入另一个约束系数，本控制不采用它、不加超参搜索。

## 固定源人口、LP与FT

完全复用上一控制服务器已冻结的5496训练RGB／687parent及source-ID292／73layout、源OOD220／55layout：两seen针型训练、metadata字典序第三种留出。无需新factor扫描、source名单、RGB、标签、seed或权重下载，旧14494／原val1506不读，不改变旧split。数据仍nativeROI→LINEAR512／ImageNet，output128仅API，原photo及geometryFalse/pairNone。

fresh既有cached完整MNV4/GAP1/semantic1280/newLinear1280→1 std.001bias0，FP32sigmoid。LP先冻结整个encoder并eval（包括77BN及其affine/buffers），只用训练5496图各一次、同原photo、loader seed20262020、不shuffle提取detachedF16特征。cache仅server device内存，不保存或导出。原point在全5496特征上固定100次AdamW3e-4/wd1e-4/clip5/defaultAMP尝试，SmoothL1beta.05、pointF16/scalarF32；每次forward5496，记录updates/AMPskip，不按loss改次数、不读验证。100是本控制资源上限，未声称LP收敛或论文默认值。

继承同一头，丢弃LPoptimizer/scaler；恢复fresh初始化后记录的随机状态（只恢复RNG、不重置学到的头），整个CNN和point恢复可训、BN恢复正常source更新。fresh全参数AdamW3e-4/wd1e-4/clip5/defaultGradScaler、同seed20262020，5完整FT epoch，train64末56、workers4、每图每E一次、无中途验证／scheduler／best选择／延长。430次全CNN反传与前一控制相同；额外LP计算不当免费，也不把结果归因于已测量的feature distortion。

## 门槛与计算成本

lastFT5只评同292行ID一次。原constant0.5双门槛完全保持：mean和ceil30 tail各≤同人口constant的一半，失败按error1保留分母；未过正常停止、OOD零RGB/forward。通过后同lastFT5源OOD220／4calls一次，真实Industrial/RF全0，不补评旧baseline的源OOD。只有原门槛通过、IDmean不高于directFT5的6.13280078773%且tail低于25.66999633486%，才保留此有限warmstart候选；失败不扫LP次数、学习率、seed、cohort或阈值，不推翻全部LP-FT理论。

正式LPcache为5496CNN图／86calls／0CNNback；LPpoint为100calls／549600次cache特征呈现／100point-onlyback。FT为27480图／430CNN和pointcalls／430CNNback。加ID后CNN33268图／521calls，point577372呈现／535calls；若过门槛加OOD，则CNN33488／525，point577592／539。总backward530=100point-only＋430fullCNN；各阶段updates＋AMPskip分别100／430。源uniqueRGB仍5496，cached feature呈现不是新图或CNNforward。

## 新检查及执行边界

只新增CPU阶段冻结/恢复、梯度模式、RNG/optimizer边界、不同CNN/point计数检查；复用旧sourcepacket和gate，不重复旧selector、GT、数学、模型构造或tests。一次新source-ID-only零更新pref恰3CNN／8张不同既有图／2pointback／1CNNback／0optimizer：LP2seen original首parent slot0、FT4seen original+roll同parent slot3、ID2seen各一；pref不看OOD。LP应无encodergrad／BNcounter更新且462encoderstate保持，FT必要stem/neck/point梯度有限非零且77BN+1、eval额外0；两back间清梯度，233参数0opt不变、464strictreload、cached462初值保持、真实512/F16semanticpoint/F32scalar/IDpipeline identity。pref固定loss128/grad128，正式fresh重建，不带pref状态。

代码/协议/config/NEWCPU证据先existingmain冻结push、实际旧父子退出且GPU空后才唯一pref→LP100→FT5→IDgate→条件源OOD；不改live、不额外GPU或两个真实外域。自身module-entry计时器必须早于torch import，supervisor wall与实际phase time/peak分开，未知point-only时间不填0。完整source合同/训练arrays/特征值/RGB/weights/geometry/privatepaths留server private，公开仅规范化保存标量及metadata。失败保留partial不fresh覆盖或自动延长，完成后唯一新鲜独立CPU保存数字核对归档。

## Limitations

预训练特征适合本任务的程度仍未知；额外LP数据处理及优化是本控制的一部分，不是同总成本比较。source-ID已用于上一控制与诊断，不能称新盲测或显著性；source-OOD只是三个合成needle中的一个固定bank成员，其他layout因素混杂。源ID改善或单域结果不等于整体泛化成功，不自动升级模型、拼checkpoint或开长训练。


启动metadata先于所有output写入检查；默认不恢复已有输出。若只在snapshot部署阶段失败且全部模型stage均未开始，root可显式恢复该zero-stage bootstrap：同protocol/config/source合同、steps空、无reader/pref、旧PID退出且GPU空，先保留原queue/protocol/config/PID/部分snapshot于同output/bootstrap_failure，再继续尚未开始的启动。任何训练或pref partial均不适用，不覆盖模型、改budget或自动resume。
