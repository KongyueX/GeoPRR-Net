# GeoPRR 补充实验跨环境执行提示词

以下内容可直接复制给训练电脑上的代码代理。执行不依赖另一台电脑的目录、聊天历史或论文仓库。

---

请在 `https://github.com/KongyueX/GeoPRR-Net` 代码仓库中完成本文约定的补充实验。仓库工作目录可在任意位置；所有相对路径以仓库根目录为基准。使用这台电脑已有的训练环境和数据、权重，先核对配置，再按固定预算训练与评测。完整模型始终为三专家 GeoPRR-Net；简单双视图是消融对照，不增加第四专家。

## 1. 取得代码与确认环境

进入已有 GeoPRR-Net checkout，先检查 `git status`，在保留本机未提交工作后同步 `main`。不要创建新分支。训练机器原环境为 Windows、Python 3.11、PyTorch 2.11.0 + CUDA 12.8、torchvision 0.26.0；依赖见 `experiments/requirements-training.lock.txt`。优先复用已有 `.venv\Scripts\python.exe`。只有环境缺失时才按 README 创建，不重装已能工作的环境。

```powershell
git pull --ff-only origin main
$Python = '.\.venv\Scripts\python.exe'
& $Python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
& $Python -m pytest -q test/test_revision_cache.py test/test_revision_controls.py test/test_revision_full_training.py test/test_revision_diagnostics.py test/test_revision_oof_heads.py test/test_revision_raw_baseline.py
if (-not (Test-Path experiments/configs/revision_20260922.local.json)) {
    Copy-Item experiments/configs/revision_20260922.example.json experiments/configs/revision_20260922.local.json
}
```

本次目标是补足精度与扰动鲁棒性的匹配证据，不要求 GeoPRR 必须胜出。保留失败行、全部预定条件和三个 seed，不根据测试集排序追加调参。

先阅读 `refine-logs/REVISION_EXPERIMENTS_20260922.md` 和 `refine-logs/REVISION_EXPERIMENT_TRACKER_20260922.md`。已发布的 full 数值及其训练预算可对照 `docs/data/geoprr_shared_full_20260911/protocol.json`、`training_stages.csv`；实际 checkpoint 构造、训练日志和调用入口决定运行细节，不能仅从当前代码默认值推断。

## 2. R0：填写本机资源与实际配置

在新建的 local JSON 中填写本机路径，不把其他电脑的绝对路径照搬过来。JSON 里的 Windows 路径可写正斜杠，例如 `D:/pointer_read/...`。留空或 null 的值必须从本机日志、checkpoint 或实际入口恢复。

- `source_train_manifest`：官方 SyncG/train 16,000 张源图。需要重新生成清单时，使用 `python -m experiments.datasets --help` 中的 SyncG official train/test 入口；不能用 14,442、12,818 或 3,182 张内部子集代替 full 源池。
- `evaluation_manifests`：官方 SyncG/test 4,000 张/145场景，RF100-VL 151张/35来源组，Industrial 1,395张/52采集组。固定三个 seed 为 20262020、20262021、20262022。
- 每个 seed 的 `foundation` 与 `source_r2mt`：该 seed 实际完整源池 foundation 和专家 bank。保留 bank 依赖的原 checkpoint 路径，现有 loader 会沿其来源链加载；不能只改 metadata 的 seed。新增 warm 和 full 入口核对实际 foundation 张量的一致性。
- `warm.manifest`：三个 seed 共同的 warm 源训练子集；已发布记录为 12,818 张、10轮，仍须恢复对应清单和真实优化配置。填写实际 learning rate、weight decay，不猜默认值。源 bank 若不依赖 warm 可复用；受 warm 改变影响的 polar、缓存、router 及相关后续 head 必须更新。
- `smooth_max_temperature`：stage-two smooth-max 的实际温度，不是 `1/alpha=0.01` 的增益温度。
- `availability_policy`：恢复正式入口后明确填写 `image_only` 或 `clean_disabled`。前者仅由 SARN 接受与模型内部有效性检查决定；后者复现已有调用方在 clean 条件禁用归一化关系的行为。后者使用条件身份，必须如实披露。相同实验块的训练缓存、full 与所有控制评测必须采用同一政策；如果决定改为纯图像策略，要将其作为明确实现变化重新评测受影响结果，不能静默混合旧预测。
- 核对各阶段增强生成器、强度、概率和总优化暴露；关系模块宽度、内部 gate、几何预算；Industrial 外层训练/内层验证/外层测试组及选 epoch 日志。

```powershell
.\experiments\run_revision_experiments.ps1 -Config experiments/configs/revision_20260922.local.json -Phase Audit
```

Audit 只读取并报告已有字段及缺项，不会替代实际配置核对。某个必需记录缺失时，明确列出对应文件/字段，并继续其他不依赖它的工作；不能把计划写成已运行。

### 数据清单格式

SyncG 使用 `experiments.datasets` 生成的原始官方 manifest，需包含 `dataset=SyncG` 和对应 `split`。程序以 `metadata.scene_name` 的场景名作为145组统计单位，不以仪表类型×场景的组合冒充145组。

外部队列使用逐行 JSONL，每行包含：

```json
{"sample_id":"stable_image_id","group_id":"actual_acquisition_or_source_group","image_path":"relative/to/manifest/roi.png","normalized_target":0.42}
```

`normalized_target` 必须来自该队列的真实标签/既定几何目标转换，不能用任何模型预测代替。图片若不是已裁好的 ROI，额外提供像素坐标 `dial_bbox:[left,top,right,bottom]`。沿用本机已经用于正式结果的固定图像与组清单；转换格式不改变成员或标签。RF100 的端点裁剪规则沿用现有评测。原始图像按同一六条件生成器处理，不重新生成另一套扰动。外部数据集不得用于 source 模型训练或选模。

## 3. E14：独立 warm 与 full 下游链

```powershell
.\experiments\run_revision_experiments.ps1 -Config experiments/configs/revision_20260922.local.json -Phase Warm
.\experiments\run_revision_experiments.ps1 -Config experiments/configs/revision_20260922.local.json -Phase Full
```

Warm 为每 seed 从自己的 foundation 拟合实际被使用的 angular_no_aux 初始化。Full 显式选择 `--population-mode official_full`，polar和最终router均使用官方16,000源图，polar 10轮，router 40+120轮，六条件缓存96,000行。它同时保存full与结构变体，并输出几何×路由的固定权重单元。Cache/Evaluate还会刷新SyncG上的结构变体，并纳入同一汇总。source专家bank的参数来源由R0确认，不在此处虚构其训练过程。

每阶段支持 `-Seeds 20262021,20262022` 只执行指定 seed。符合相同配置且独立的现有20262020链可复用，但须提供完整的实际记录；不能只复用共享初始化后改 seed 名称。运行器不覆盖已有输出目录，先检查已完成的日志，再选择需要执行的阶段或新的输出目录。

## 4. E20/E27：匹配缓存上的门控和双视图对照

```powershell
.\experiments\run_revision_experiments.ps1 -Config experiments/configs/revision_20260922.local.json -Phase Cache
.\experiments\run_revision_experiments.ps1 -Config experiments/configs/revision_20260922.local.json -Phase Controls
.\experiments\run_revision_experiments.ps1 -Config experiments/configs/revision_20260922.local.json -Phase Evaluate
.\experiments\run_revision_experiments.ps1 -Config experiments/configs/revision_20260922.local.json -Phase Summarize
```

- E20线性gate保留同一shared projection与220维输入，只将最终增益MLP改为 `LN(220)→Linear(220,2)`；标量gate为 `LN(11)→Linear(11,96)→GELU→Linear(96,2)`。两者沿用full的两阶段损失、预算、EMA、先验与alpha。
- E27为冻结共享B0上的 `LN(2560)→Linear(2560,320)→GELU→Linear(320,1)→Sigmoid`。40固定缓存轮，AdamW lr3e-4/wd1e-4/batch512/cosine/L1，使用terminal checkpoint；不增加第四专家，不扩大为端到端训练。
- full、base-only、polar-only、relational-only、uniform、fixed-prior与三个训练控制使用相同 evaluation cache。训练缓存严格为source_train范围；不得拿评测缓存训练或选epoch。
- gate使用EMA，双视图使用terminal，在查看测试结果前固定，不在两种权重间择优。
- 评测输出sample/group/condition/seed/target/prediction/error/status的CSV及输入政策侧车。参数量写入训练配置；完整图像到读数的延迟仍需在同一GPU上实际测量，不能把缓存head时间当完整流水线延迟。

## 5. 条件触发的C4/C13

只有R0说明确实需要补时执行，不加入默认批次。

C4使用 `experiments.revision_raw_baseline train`：官方16,000源图、60个source-equivalent epochs，每轮每图一次，均匀选一个现有六条件。它固定Raw B0、60轮/30+30学习率、Smooth-L1 beta0.05和相同测试像素。使用 `--help` 给出的实际参数完成三个seed，之后由其 `evaluate` 在三个完整队列输出同格式CSV，再和匹配full进行统一汇总。不要把96,000行×60轮称作同样的训练暴露。

C13使用 `experiments.revision_oof_heads`，输入本机实际冻结特征缓存及共同 `group_to_fold` 映射。每个外层k为测试、(k+1)mod5为内层验证、其余三折拟合；128/64 head、dropout0.1、最多100轮、patience20，按验证Acc5/Acc2/负NMAE选epoch，平局保留早epoch。loss、activation、lr、weight decay和batch size必须来自实际记录；可选标准化只在当前训练组拟合。按同一配置分别运行四方法×三seed，输出完整OOF与组划分记录。特征缓存为torch字典：`dataset="industrial"`、`scope="target_adaptation"`、`features`（N×D）、`target`（N）、逐行 `sample_ids/group_ids/condition`、`seed`、`method` 和 `feature_definition`；必须包含1,395图×6条件、52组，同图标签与组保持不变。不能假设所有方法的head输入都是同一维度或均为raw pooled feature。无真实特征导出记录时先完成对应本机导出，不以猜测取代。

调用示例（尖括号内容替换为R0实际配置）：

```powershell
& $Python -m experiments.revision_raw_baseline train --manifest <official_train.jsonl> --seed 20262020 --device cuda:0 --output <C4/seed20262020/terminal.pt>
& $Python -m experiments.revision_raw_baseline evaluate --checkpoint <C4/seed20262020/terminal.pt> --manifest <industrial.jsonl> --cohort industrial --device cuda:0 --output <C4/seed20262020/industrial.csv>
& $Python -m experiments.revision_oof_heads --cache <features.pt> --folds <group_folds.json> --output <C13/method/seed20262020> --method <actual_method_id> --seed 20262020 --lr <actual_lr> --weight-decay <actual_wd> --batch-size <actual_batch> --loss <l1_or_smooth_l1> --activation <gelu_or_relu> --device cuda:0
```

`group_folds.json`为 `{"group_to_fold":{"group_a":0,"group_b":1}}` 这种结构，实际文件须覆盖全部52组并具有五个非空fold。使用smooth_l1时另外明确`--beta`；仅在原协议使用标准化时添加`--standardize`。不要将示例中的两个组当作完整划分。

## 6. 汇总、诊断与交付

`revision_diagnostics summarize` 校验三seed、六条件、方法间样本/组/目标一致；保留失败误差1.0，准确率中判错；三seed mean/sample SD和先平均预测的等权集成分别报告。配对group bootstrap为20,000次，SyncG/RF100/Industrial分别使用20260907/20261007/20261107，条件化于已拟合模型。增加C4/C13的CSV时用相同命令显式指定文件，不改已有比较分母。

`revision_diagnostics audit --checkpoints <evaluation_cache.pt> --output <diagnostics.json>` 汇总routing entropy、平均专家权重、集中比例以及与同一裁剪路由目标相比的矩残差。极坐标熵不是路由熵。三个逐专家及逐条件指标随 Evaluate/Summarize 自动生成，图3区间只对预测发生变化的比较刷新。

Industrial native专项仍暂停；不要把离线native数值写成新主结果。六条件中的clean行保留在既定完整分母内。现有公开结果是参考基线，不是这次运行的产物；新实验完成前不覆盖它们。

每个里程碑更新tracker，记录配置、依赖、样本/组、seed、epoch、checkpoint选择、耗时、失败与完整预测。工程成功标准是可复算和协议匹配，负结果同样交付。输出保存在配置指定的artifacts目录，最终提供汇总表、实际命令、资源/配置缺项和待回填论文的位置。本任务不依赖paper仓库，不编译LaTeX或渲染Word；不要将本地图片、权重、大缓存或带私人路径的运行配置提交到Git。
