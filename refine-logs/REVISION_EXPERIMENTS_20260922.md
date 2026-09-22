# GeoPRR 论文补充实验清单

目标是在统一 full 方法下验证读数精度和扰动鲁棒性。完整模型仍为几何／基础、极坐标、关系三个专家。现有实验数据保留；新增训练按下列固定范围执行，结果不预设方向。

## 已固定的评测口径

| 数据集 | 独立图像 | 组 | 每种子六条件记录 | 作用 |
|---|---:|---:|---:|---|
| SyncG official train | 16,000 | 145 | 96,000条路由训练缓存 | 唯一源域训练池；预训练初始化阶段按已记录固定样本清单 |
| SyncG official test | 4,000 | 145 | 24,000 | 图像隔离的源域精度与扰动鲁棒性 |
| RF100-VL | 151 | 35 | 906 | 无目标域拟合的公共数据迁移 |
| Industrial-1395 | 1,395 | 52 | 8,370 | 六条件冻结迁移与单独报告的目标监督OOF；native专项暂停 |

三个source seed固定为20262020、20262021、20262022。六条件为clean、blur_moderate、blur_severe、perspective_moderate、perspective_severe、combined_severe。各方法使用相同ROI、样本目标、条件像素及分组，保留失败样本和完整分母。不得将六个条件当成六倍独立采集样本。

核心主张只有两项：一是结构化候选与条件路由改善受控扰动下的读数；二是该读数器能够迁移至源训练集合之外的真实图像。Industrial native的结果核对及文稿修改暂缓；本轮保留既有六条件评测口径。

## 必须完成或核对的五个工作块

| ID | 对应意见 | 工作 | 规模 | 完成证据 |
|---|---|---|---|---|
| R0 | 4、13、21–23、25 | 找回实际full训练／评测入口、每seed checkpoint construction、增强配置、OOF内层组分配及SARN availability规则 | 配置读取和定向复现，无需新增训练 | 具体配置表、组分配表、可执行入口；不得用默认值代替实际配置 |
| E14 | 14 | 各source seed独立warm初始化，重跑受影响polar、缓存和router依赖链 | 通常新增2份warm及2条依赖链；若共同设置改变则3条 | 每seed独立初始化来源、训练日志、完整评测；受影响适配和消融同步更新 |
| E27 | 27 | 简单双视图回归头并入消融；并列geometry/base-only与full | 3个控制头；另两行可由完整模型候选汇总 | 控制头配置／权重、逐行预测、三seed及逐条件结果 |
| E20 | 20 | 同候选缓存上的线性增益head和仅11标量的MLP gate；补固定等权推理 | 2种gate×3seed=6个router拟合；等权无需训练 | 匹配缓存、固定训练设置、逐行输出、性能与权重诊断 |
| C4/C13 | 4、13 | 只有R0无法证明所需条件或确认条件不满足时执行增强匹配／OOF重做 | C4最多先补Raw B0 3次；C13最多4方法×3seed×5fold=60个小head | 明确的实际增强暴露和内层选模隔离；不得把计划当运行证据 |

### R0 配置恢复

先定位项目已授权GPU环境及对应full入口。当前本地若干训练入口调用内部场景划分，不应仅通过替换路径就假定能够训练完整16,000张源图。

需恢复的内容：

- warm初始化的训练样本、迭代预算和seed；polar及最终router的依赖关系。
- Raw CNN实际使用的增强生成器、强度、采样概率与优化步数；六条件缓存不自动等于CNN训练增强分布。
- relational heads和内部risk gate的latent／hidden宽度、残差上界、混合强度；geometry token宽度、progress-mixing开关及预算增益；stage-two的smooth-max temperature。
- Industrial每外层fold的train／inner-validation／outer-test组，以及选epoch日志。外层预测记录中的outer_fold不能代替内层分配记录。
- 实际图像入口从SARN接受标记到模型relation availability的计算。部分本地调用方存在条件相关覆盖，不能推断full采用它，也不能在未核对时称全部入口等价。

配置未找到时，继续可独立完成的实验代码和现有预测分析；把缺少的入口／权重／清单具体列出，不猜测后声称复现。

### E14 独立warm初始化

保持共同训练样本、模型定义和优化预算固定，仅改变随机种子。每个seed使用自己的源域foundation和warm initializer。若20262020现有整条训练链完全符合共同设置，可复用；新增20262021和20262022的warm训练。

重新执行受影响的polar训练、候选缓存、最终router以及使用变化候选的消融router。与warm无依赖的已训练基础模块可复用。目标域适配若使用变化的特征或读数，也应重新拟合相应head。不能只修改checkpoint路径或seed标签而复用依赖原polar的缓存。

终点是三组独立初始化的完整预测，分别在三个评测队列的六条件清单上汇总。Industrial native专项保持暂停，不列为本轮训练链验收条件；用户恢复该项后再处理。工程验收看执行完整与数据一致性；科学判断看误差、配对差值和波动，不要求所有seed都改善。

### E27 基础专家与专家组合消融

只新增一个控制头，不增加第四专家。固定每seed的共享EfficientNet-B0，缓存raw与support-normalized的1280维全局特征。控制头为：

`LayerNorm(2560) → Linear(2560,320) → GELU → Linear(320,1) → Sigmoid`

三个拟合共享16,000张源图、同一SARN与六条件，训练仅使用源标签。控制头的拟议固定设置为AdamW、lr=3e-4、weight_decay=1e-4、batch=512、40个固定缓存epoch、cosine schedule、归一化L1目标、terminal checkpoint；在第一次查看测试结果前固定，不执行测试驱动搜索。这是新增实验设置，不写成已完成配置。

同一消融表中加入：Simple dual-view only、Geometry/base only、Full MoE；保留现有no-geometry、no-relational和fixed-mixture行。Geometry/base only读取完整模型的基础候选，不声称它是另一个从头训练的网络。Simple dual-view only是单独拟合的控制头，不能把不同训练目标隐藏成严格单层替换。

报告三seed NMAE、Acc@5%、逐条件差异、参数量和实际完整读数延迟。约4.83M只能作为规划估算。若普通双视图接近或优于full，如实保留，收紧复杂结构必要性主张，不继续增大或削弱baseline以改变排序。

### E20 简单gate控制

同一full候选库、每seed96,000条源缓存、相同两阶段损失与40+120阶段预算，冻结全部候选模块。

1. Linear gain head：保留与full相同的投影与220维输入，将最终非线性MLP改为`LayerNorm(220) → Linear(220,2)`。
2. Scalar-input MLP：只输入候选均值、候选差、两分支读数、availability、polar concentration和polar entropy，共11维；使用`LayerNorm(11) → Linear(11,96) → GELU → Linear(96,2)`。
3. Uniform与fixed prior：直接从同一候选行计算，不训练。

学习型gate保留0.10 tanh、alpha=100、prior=(.50,.25,.25)、同optimizer／EMA／梯度裁剪及参数初始化规则。线性head仍有共同非线性特征投影，其结论只针对最终增益映射的非线性。新增gate不能读取测试标签、真实扰动参数或额外标注。

### C4/C13 条件触发的工作

**C4增强匹配**：先从R0记录判断。若确实需要补，先针对最强Raw EfficientNet-B0，使用同一六条件实现、相同源图和seed。以优化步数固定预算，例如60个source-equivalent epochs，每次原图暴露均匀抽一个条件，保持总原图暴露为16,000×60，不把96,000行的60epoch误称相同预算。记录各阶段分布差异，不声称CNN与多阶段MoE的全部优化过程相同。

**C13 OOF**：实际存在独立内层组时，恢复并披露即可。若需重做，统一五外层fold；测试fold k、验证fold(k+1)mod5、其余三fold训练。保持已有128/64 head、dropout .1、最多100epoch、patience20和真实优化设置；选择仅用内层(Acc@5,Acc@2,-NMAE)字典序，平局取较早epoch。数据相关归一化仅用训练组。所有方法和seed共享划分，完成五折后拼接OOF。无需额外完整内层五折搜索。

## 无需训练的汇总与文稿工作

| 工作 | 当前状态 | 下一步 |
|---|---|---|
| Industrial native七指标及52组20,000次bootstrap | 暂缓；离线汇总文件已保留 | 待用户恢复该项后再核对，不纳入本轮论文修改 |
| Figure3场景bootstrap置信带 | 现有结果已完成 | 复用`data/revision_20260922/figure3_*`；使用145场景、同组配对、完整分母；受后续checkpoint变化影响时刷新 |
| alpha尺度与集中度 | 尺度解释、六尺度扫描和集中度已写入论文；routing entropy待汇总 | 保持post-hoc固定模型分析；逐样本routing entropy需补齐逐行权重 |
| 各专家、uniform、fixed、full逐条件读数 | 部分诊断已有，待补齐逐行候选与权重缓存 | 优先复用缓存，补Acc@5%、均值及entropy，不需重新训练专家 |
| 末端projection数值差与成本 | 部分诊断已有，待统一 | 同checkpoint比较裁剪路由目标和投影均值；可选耗时仅在实测后报告 |
| 结构、损失、SARN复现说明 | 定义、公式和算法已补入论文；实际配置待R0确认 | 补实际模块宽度／预算、smooth-max温度及系数记录、增强暴露、内层组分配和availability入口；不用默认值代填 |

## 实现入口

跨环境配置模板为 `experiments/configs/revision_20260922.example.json`，Windows 分阶段入口为 `experiments/run_revision_experiments.ps1`。E14 使用 `revision_full_training warm-polar` 和 `train_unified_pointer_reader --population-mode official_full`；E20/E27 使用 `revision_controls`；缓存及统计使用 `revision_cache` 和 `revision_diagnostics`。C4/C13 仅在确认需要时分别调用 `revision_raw_baseline` 和 `revision_oof_heads`。全部路径、字段格式和命令见同目录的执行提示词。

这些入口是新增补充实验的实现，不代表缺失结果已经产生；实际入口与参数仍以R0恢复的记录为准。

## 运行顺序和停止条件

R0 → E14 → 刷新受影响预测／缓存 → E27与E20 → 需要时C4/C13 → 全部统计与论文表图同步。

遇到样本清单、目标、分组、条件实例或输入契约不一致时，先修正该具体问题；不要带着不一致结果继续比较。一次实验按预定预算完成即结束，不根据外部测试输赢追加seed或调参。完整输出所有预定方法和seed的不利结果。

训练总量下限为2条受影响warm链、3个双视图head和6个gate拟合；实际可复用量由R0确定。GPU小时不从缺少测时的阶段猜测，首次代表性运行后按实测单元耗时估算剩余工作。无需租用新GPU或扩大数据集。

## 结果交付

每个运行保存实际配置、seed、checkpoint来源、训练阶段及迭代数、所用样本与组、选择规则、逐行输出、耗时和失败记录。统一字段至少包含method、seed、sample_id、group_id、condition、target、prediction、absolute_error、status；适配另含outer_fold。候选诊断另含三候选、三权重和availability。

逐seed先算指标再取mean和sample SD。bootstrap按组抽样，同一次组抽样用于所有方法和seed；RMSE、Acc和分位数先在每seed的重采样集合计算，再跨seed平均。集成先平均同一行预测再算误差，单列报告。正文不出现尚未执行实验的性能数值。
