# 务实的论文实验方案

截至 2026-07-24，主实验、六组控制退化、VDN 三种子对比、概率方向、进度校准、最终选择
路由和统一复杂度基准均已
完成。聚合结果、置信区间与可安全主张的结论见
[`../docs/FORMAL_RESULTS_CN.md`](../docs/FORMAL_RESULTS_CN.md)；
本文件继续作为复现实验协议和命令说明。

标量读数主实验只使用两个正式数据集：

- **SyncG**：只用官方 `train` 微调指针分割、训练选择性残差校正器与独立方向头，并使用官方 `test` 做同域测试。
- **RPM-10K**：只做冻结模型的零微调真实图像外部测试，不用于选特征、阈值或超参数。

此外只增加一个**辅助部件外测**：Pointer-10K 官方 test 用于比较真实图像上的单指针方向，
不训练、不微调、不标量读数，也不与 RPM-10K 重复承担最终读数结论。这样仍然是
“SyncG 训练/同域测试 + RPM-10K 最终读数外测 + Pointer-10K 方向部件外测”，不是让三套
模型在五套数据库上雨露均沾。

仓库原有的 30+8 张图只适合作为补充案例，不并入主表，也不作为“独立数据集”反复调参。
这样既避免五套数据集雨露均沾，也能把核心论点讲清楚：精确但易失败的掩码表示和高覆盖率
方向表示能否通过选择性路由互补，以及残差校正在退化场景中能否控制负迁移。

官方下载入口：[SyncG 代码与数据说明](https://github.com/SKYWOWDYH/SyncG)、[SyncG 数据](https://huggingface.co/datasets/YihengDeng/syncG)、[RPM-10K / DialBench](https://github.com/Event-AHU/DialBench)、[Pointer-10K / VDN](https://github.com/DrawZeroPoint/VectorDetectionNetwork)。固定提交中的 SyncG 数据卡标注为 CC BY 4.0，Pointer-10K 数据标注为 CC BY-NC-SA 4.0，使用及论文中应保留数据集归属；RPM-10K 仓库当前把数据许可标为 TBD，投稿或再分发前必须再次核对。本仓库只提供下载/解包脚本，不分发数据。

外部同类模型不能由内部变体替代；VDN 等方法的可复现性和公平接入规则见
[`../docs/BASELINE_AUDIT_CN.md`](../docs/BASELINE_AUDIT_CN.md)。

## 内部方法（消融表）

所有读数方法共享同一次前端推理缓存。分割网络只允许在 SyncG `train` 内微调；进入读数实验后，检测、校正、分割和起终点网络全部冻结。

| 名称 | 含义 |
|---|---|
| Original Transformer | 项目原始的分割掩码 + meter transformer 读数基线 |
| Geometry-v1 | 直线拟合后取最远针尖 |
| Geometry-v2 | 外侧候选点稳健方向投票 |
| Mean Fusion | v1/v2 简单平均，保留为消融 |
| Quality-weighted Fusion | 使用轴线一致性、针尖支持、方向投票集中度与正反侧分离度估计质量并加权 |
| Residual without Gate | 在质量加权结果上无条件加残差 |
| Ours-mask | 质量加权 + 归一化残差 + 不确定性/学习门控 |
| Direction-v1 | 独立支点热图 + 全局方向向量，不使用指针分割 |
| Hard route-v1 | Ours-mask 有输出时保留，仅在硬失败时调用 Direction-v1 |
| Quality route-v1 | 只在 SyncG train 跨模型 OOF 上学习的第一代质量切换 |
| Probabilistic vector | 支点热图 + 直接方向 + 72-bin 圆周分布 + 角方差 |
| Calibrated vector | 在 Probabilistic vector 上增加透视感知 angle-to-progress 残差校准 |
| Ours-final | mask / Calibrated vector 的 grouped-OOF 安全选择路由，保留硬失败规则 |

历史 JSON 中字段名 `Ours` 对应这里的 `Ours-mask`；保留字段名是为了不改写已经冻结并签名的
预测缓存。

残差目标为 `(GT - weighted_fusion) / (scale_end - scale_start)`，因此不同量程可以共用模型。残差与门控采用按 `group_id` 划分的**嵌套交叉拟合**：外层测试组既不会进入残差模型，也不会参与残差裁剪分位数、门控模型及其训练标签的构造；内层残差裁剪同样只由对应内层训练组确定。最终部署模型的裁剪值才由完整 SyncG train 冻结，门控阈值只在其外层 OOF 结果上确定。

门控特征全部可在运行时获得，包括两种几何估计的分歧与置信度、mask 轴线统计、分割概率的最大值/P99/均值/前景比例、起终点分支和表盘椭圆代理；面积按裁剪面积归一化，距离按短边归一化，不依赖固定像素分辨率，也不读取真值或数据集名称。RPM-10K 上的低分割置信度因此只能触发保守回退，不能参与重新拟合。

## 0. 固定环境与下载数据

当前锁文件已在 Windows、RTX 4060、Python 3.11.15、PyTorch 2.11.0+cu128 上验证。使用 `uv` 创建隔离环境：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe `
  -r experiments\requirements-training.lock.txt
```

SyncG 仓库共 24.3 GB，其中 `scene_file/` 是重新生成合成数据所需的 HDR 环境，并非训练/评测必需。下面的脚本只下载官方提交中 5 个分卷及 train/test 划分文件（约 20.7 GB），逐文件验证大小和 SHA-256，并可直接解压。正式 manifest 与训练入口还会核对官方 16000/4000 个样本 ID 的固定哈希，不能用“数量相同但内容不同”的目录冒充固定发布版：

```powershell
# 默认通过本机 FlClash 的 SOCKS5 端口解析签名地址，并经 HTTP 代理单连接传输。
# 某些代理/CDN 会对多连接限速；确认链路支持并发后可显式传 -Connections 4。
.\experiments\download_syncg.ps1 -Extract

# 网络可直连 Hugging Face 时：
.\experiments\download_syncg.ps1 -SocksProxy "" -DataProxy "" -Extract

# 多链路下载时可让另一个终端只负责指定分卷；不要让两个进程下载同一分卷：
.\experiments\download_syncg.ps1 -IncludeFiles syncG.z04
```

脚本需要命令行中的 `curl.exe`、`aria2c.exe` 和（解压时）`7z.exe`。下载支持断点续传；已通过校验的分卷会自动跳过。大文件传输超过 CDN 签名有效期时，脚本会自动刷新签名并继续同一断点。数据、环境和实验产物均位于 Git 忽略目录，不能提交到代码仓库。

RPM-10K 只下载正式外测所需的标签和图像，不下载 8730 张训练图：

```powershell
# 默认使用本机 SOCKS5 代理；会校验固定 test.json，并恢复未完成图像。
python -m experiments.download_rpm10k_test

# 网络可直连 Google Drive 时：
python -m experiments.download_rpm10k_test --proxy ""
```

Google Drive 匿名嵌入页只枚举前 5500 个文件。脚本通过隔离的无登录 Chrome 会话捕获公开分页请求，再分页列出官方 10730 张图；不读取浏览器用户配置、账号或 Cookie。

两套数据准备完成后，可用一个可恢复脚本依次执行清单、分割微调、预测缓存、嵌套 OOF、冻结外测和论文表格：

```powershell
.\experiments\run_paper_experiments.ps1

# 附录还需要四组廉价的特征消融时：
.\experiments\run_paper_experiments.ps1 -RunFeatureAblations
```

预测缓存已存在时脚本使用签名校验后续跑；完整的分割训练已有 `summary.json` 时，会先复核固定数据身份、训练参数、检查点哈希和内部签名，并逐张量确认 `released_calibrated.pt` 与原发布权重一致，再决定复用，不会把 `--limit` 诊断产物混入正式流水线。

## 1. 生成统一清单

SyncG 在 Hugging Face 上以分卷压缩文件发布；下载全部 `syncG*` 分卷并解压后，目录应包含 `annotations/train|test`、`images/train|test` 与 `masks/train|test`：

```powershell
python -m experiments.datasets syncg `
  --root D:\datasets\SyncG `
  --split train `
  --output artifacts\manifests\syncg_train.jsonl

python -m experiments.datasets syncg `
  --root D:\datasets\SyncG `
  --split test `
  --output artifacts\manifests\syncg_test.jsonl
```

RPM-10K 的完整测试集含多表盘文本答案和 `others` 网络表盘，而本项目一次只输出一个指针的标量读数。预先固定的适用域协议只保留六类主表盘中标量读数位于 `[0, range]` 的样本：2000 张中保留 1797 张，排除 194 张 `others`、7 张非标量答案和 2 张越界标签。该规则只读取标签结构，不读取任何模型输出；生成的 `*.protocol.json` 会记录全部计数。因此论文必须写作 **RPM-10K single-pointer subset**，不能冒充完整官方 2000 张榜单。

```powershell
python -m experiments.datasets rpm10k `
  --root datasets\RPM10K\images `
  --labels datasets\RPM10K\labels\test.json `
  --output artifacts\manifests\rpm10k_single_pointer_test.jsonl
```

## 2. 只在 SyncG train 微调分割前端

生产路径中的 U2NetP 原先把 256×256 Letterbox 输出直接拉伸回表盘大小，没有先移除补边；当前实现已修复这一坐标映射。微调进一步使用真值表盘框裁剪 SyncG 图像与指针掩码，以 `gauge_type::scene_name` 为组从官方 `train` 内部留出 10% 验证集。官方 `test` 不参与早停、损失选择或二值化阈值选择。

```powershell
python -m experiments.train_syncg_segmentation `
  --root datasets\SyncG `
  --output-dir artifacts\runs\syncg_segmentation `
  --device cuda `
  --epochs 15 `
  --batch-size 8 `
  --seed 20260720
```

`best.pt` 同时封装模型参数和在 train-validation 上、经过生产一致的最大连通域后处理后冻结的概率阈值，推理器会自动读取。为避免组件对比把“微调权重”和“阈值校准”混为一谈，训练脚本也会生成 `released_calibrated.pt`：其权重与发布模型逐参数相同，只在同一个 train-validation 上校准阈值。训练裁剪会小幅随机扰动表盘框边界，以模拟检测框偏差；BCE/Dice 深监督和验证指标均只在移除 Letterbox 补边后的真实内容区计算。混合精度训练只在网络前向使用 FP16，损失使用 FP32 logits 计算，并把 GradScaler 初始尺度固定为 512；每轮记录成功及跳过的优化器步数，防止数值溢出被静默忽略。下面的组件消融使用官方 test 的真值表盘框，只报告分割上限，不混入端到端主表；测试集不会重新搜索阈值：

```powershell
python -m experiments.evaluate_syncg_segmentation `
  --root datasets\SyncG `
  --checkpoint released=artifacts\runs\syncg_segmentation\released_calibrated.pt `
  --checkpoint finetuned=artifacts\runs\syncg_segmentation\best.pt `
  --output artifacts\runs\syncg_segmentation\syncg_test_metrics.json `
  --device cuda
```

这是唯一新增的前端训练实验；不训练 RPM-10K，也不同时展开检测、关键点、校正四条训练线。
为检验合成分割微调是否在真实域发生负迁移，完整流水线只额外对 RPM-10K 外测缓存一次 `released_calibrated.pt` 前端，并生成 `frontend_transfer_table.md`。该表仅比较 released / SyncG-finetuned 分割搭配 Original Transformer 与 Quality-weighted Fusion；脚本强制两份缓存除分割检查点外完全同签名。它是冻结迁移诊断，不能根据 RPM 标签回头选择前端或阈值。

## 3. 前端只运行一次

主实验使用完整图像和固定 `correction_mode=off`，避免把校正参数调优混进读数算法比较：

```powershell
python -m experiments.collect_predictions `
  --manifest artifacts\manifests\syncg_train.jsonl `
  --output artifacts\predictions\syncg_train.jsonl `
  --segmentation-weights artifacts\runs\syncg_segmentation\best.pt `
  --correction-mode off `
  --device cuda

python -m experiments.collect_predictions `
  --manifest artifacts\manifests\syncg_test.jsonl `
  --output artifacts\predictions\syncg_test.jsonl `
  --segmentation-weights artifacts\runs\syncg_segmentation\best.pt `
  --correction-mode off `
  --device cuda

python -m experiments.collect_predictions `
  --manifest artifacts\manifests\rpm10k_single_pointer_test.jsonl `
  --output artifacts\predictions\rpm10k_single_pointer_test.jsonl `
  --segmentation-weights artifacts\runs\syncg_segmentation\best.pt `
  --correction-mode off `
  --device cuda
```

中断后加 `--resume` 可继续。`--use-manifest-crop` 是使用真值框的诊断上限，只能放附录，不能与完整管线结果混在主表。

每个预测 JSONL 旁边会生成 `*.meta.json`，记录 manifest 及其固定发布协议快照、四个权重、核心源码和推理配置的 SHA-256。`--resume` 会核对该签名，避免把不同前端或不同协议的结果混进同一缓存；正式拟合还会再次要求协议为已验证的 SyncG train、缓存恰好 16000 行且样本 ID 哈希一致。

## 4. 仅在 SyncG train 拟合

```powershell
python -m experiments.selective_experiment fit `
  --train-predictions artifacts\predictions\syncg_train.jsonl `
  --output-dir artifacts\runs\syncg_full `
  --feature-set full `
  --folds 5 `
  --seed 20260720
```

输出包括：

- `calibrator.joblib`：可被 API 直接加载的残差+门控包；
- `oof_predictions.jsonl`：无训练样本自预测的 OOF 结果；
- `training_summary.json`：六方法训练诊断、门控 Brier/ECE/AUROC；
- `oof_risk_coverage.csv`：门控阈值的风险—覆盖率曲线。

## 5. 冻结评测，禁止在 RPM-10K 调参

```powershell
python -m experiments.selective_experiment evaluate `
  --predictions artifacts\predictions\syncg_test.jsonl `
  --calibrator artifacts\runs\syncg_full\calibrator.joblib `
  --output-dir artifacts\runs\syncg_test

python -m experiments.selective_experiment evaluate `
  --predictions artifacts\predictions\rpm10k_single_pointer_test.jsonl `
  --calibrator artifacts\runs\syncg_full\calibrator.joblib `
  --output-dir artifacts\runs\rpm10k_single_pointer_zero_shot
```

脚本统一报告成功覆盖率、MAE、RMSE、NMAE、NRMSE、Acc@1%、Acc@2%、逐表宏平均 NMAE、门控校准指标、修正覆盖率和负迁移率，并按 `group_id` bootstrap 计算 NMAE 及关键配对差值的 95% 区间。主表与配对 bootstrap 的归一化指标采用同一个端到端分母：无读数样本在 NMAE 中按量程内最坏误差 `1.0` 计，在 Acc 中直接算错；`successful_nmae` 和配对共同成功数等字段另行保留诊断。DialBench 官方公式为 `Ref=|y-ŷ|/Range`、`Rel=|y-ŷ|/y`，没有 10/100 截断；代码因此另存未截断的成功输出 `dialbench_ref_successful` / `dialbench_rel_successful`，并用全样本分母报告 Ref≤0.01 的 Accε 和 Rel<0.05 的 Accθ。原有 10/100 截断项仅作为给无输出分配显式惩罚的稳健性诊断，不能写成官方榜单指标。结果还包括六类表盘和八种环境条件的冻结分层。
完整入口会从两份 `risk_coverage.csv` 自动生成 `risk_coverage.png` 和矢量 PDF；绘图只读取冻结结果，不会再次拟合或调用视觉模型。

评测时还会核对训练/测试缓存的前端签名；正式结果的 `front_end_signature_verified` 必须为 `true`。只有非论文诊断才应使用 `--allow-front-end-mismatch`。

生成论文主表：

```powershell
python -m experiments.make_paper_table `
  --syncg artifacts\runs\syncg_test\metrics.json `
  --external artifacts\runs\rpm10k_single_pointer_zero_shot\metrics.json `
  --external-name "RPM-10K single-pointer" `
  --output artifacts\runs\main_table.md
```

完整流水线还会从这两份冻结缓存生成 `artifacts\runs\failure_table.md`
及同名 JSON，按表盘未检出、指针未检出、读数后端失败和运行异常归因；
它与主指标使用相同的全样本分母，不会触发第二次视觉推理。
此外，`frontend_transfer_table.md` 使用 fine-tuned 与 released 两份 RPM 缓存量化合成到真实的分割迁移；这仍是同一个外测数据集，不增加第三套数据协议。

## 6. 模糊与透视鲁棒性协议

论文主张进一步收敛为“在模糊和非常规拍摄视角下保持稳定读数”。该主张不能只依赖
RPM-10K 的人工环境标签，而采用一套固定的 **SyncG test 控制退化 + RPM-10K 真实困难子集**
协议：

- `clean`：原始图像；
- `blur_moderate` / `blur_severe`：高斯模糊标准差固定为图像短边的 0.15% / 0.30%；
- `perspective_moderate` / `perspective_severe`：虚拟平面分别作 25° / 45° 的俯仰或偏航投影；
- `combined_severe`：45° 透视与重度模糊叠加。

俯仰/偏航方向由 `seed + sample_id` 的 SHA-256 确定，同一样本在不同强度下保持相同方向，
所有方法读取完全相同的退化图像。退化只用于冻结测试；分割网络、残差模型和门控仍只在干净的
SyncG train 上训练，不能用退化 test 或 RPM-10K 回调参数。先运行 24 张端到端烟雾测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_robustness_experiments.ps1 `
  -Limit 24 -BootstrapIterations 20
```

正式运行六个固定条件（每个条件均为完整 4000 张，默认 2000 次分组 bootstrap）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_robustness_experiments.ps1
```

输出位于 `artifacts/runs/robustness/`：Markdown/CSV/JSON 主表、PNG/PDF 退化曲线，以及每个
条件的冻结指标。报告生成器强制检查六个条件样本数相同、退化源码/种子相同、校准器相同且
前端签名验证成功，并从六份同样本预测计算相对 clean 的成对 ΔΔNMAE 及分组 bootstrap 区间。
RPM-10K 的 `blur` 和 `tilted` 是多标签子集，样本可以重叠；报告另列二者交集，只作真实图像
支持证据，不应写成独立数据集。

两个主指标按样本 (i) 的量程归一化误差定义：

```text
e_i = |prediction_i - ground_truth_i| / |scale_end_i - scale_start_i|
NMAE = mean(e_i)
Acc@2% = mean(success_i and e_i <= 0.02)
```

例如量程为 0–100、真值为 60、预测为 63，则该样本的归一化误差为 0.03。`NMAE=0.10`
表示平均误差相当于量程的 10%，并不表示“准确率为 90%”；`Acc@2%=0.43` 表示 43% 的样本
误差不超过各自量程的 2%。端到端主指标对无输出样本赋 NMAE 惩罚 1.0，并在 Acc@2% 中
直接计错，同时单独报告 coverage，避免失败样本被静默删除。

报告中的 `ΔΔNMAE` 定义为：

```text
(Ours_degraded - Ours_clean) - (Transformer_degraded - Transformer_clean)
```

它回答的是“谁相对 clean 退化得更少”，与困难条件下谁的绝对 NMAE 更低是两个不同问题。
负值表示 Ours 相对退化更小；正值表示 Ours 仍可能绝对更好，但领先幅度在收窄。论文必须同时
报告绝对差值和该成对相对退化统计。

## 7. 最小消融

不再增加数据集，只重复快速的离线拟合。前端预测缓存无需重跑：

```powershell
python -m experiments.selective_experiment fit --train-predictions artifacts\predictions\syncg_train.jsonl --output-dir artifacts\runs\ablation_geometry --feature-set geometry
python -m experiments.selective_experiment fit --train-predictions artifacts\predictions\syncg_train.jsonl --output-dir artifacts\runs\ablation_no_mask --feature-set no_mask
python -m experiments.selective_experiment fit --train-predictions artifacts\predictions\syncg_train.jsonl --output-dir artifacts\runs\ablation_no_ellipse --feature-set no_ellipse
python -m experiments.selective_experiment fit --train-predictions artifacts\predictions\syncg_train.jsonl --output-dir artifacts\runs\ablation_disagreement --feature-set disagreement
```

建议正文只留三项：无质量权重、无选择门控、完整方法。其余特征消融放附录。

## 8. 外部 VDN 对比与三随机种子

VDN 官方仓库提供架构和训练代码但没有可用的官方权重。为避免许可证混入，本项目不复制其
GPL-3.0 源码，而是校验并动态加载独立、被 Git 忽略的固定提交：

```powershell
git clone https://github.com/DrawZeroPoint/VectorDetectionNetwork.git `
  artifacts\vendor\VectorDetectionNetwork
git -C artifacts\vendor\VectorDetectionNetwork checkout `
  68afe1efbdb35d3196d9a6243bfac8e5c9de5ceb
```

使用与本文相同的 SyncG train、分组 train-validation 和官方配置指定的
`resnet18-5c106cde.pth` 初始化重训 ResNet-18 + 三层反卷积双头架构。正式配置固定为
100 epoch，把官方 200 epoch 配置的 140/190 里程碑按比例缩放至 70/95，并按验证方向 MAE
选择检查点；SyncG test 和 RPM 不参与选择。注意固定提交的 `get_optimizer` 没有把 YAML 中的
`WD=0.0001` 传给 Adam，因此忠实运行的有效 weight decay 是 0：

```powershell
python -m experiments.train_vdn_syncg `
  --epochs 100 --batch-size 8 --workers 4 --seed 20260720 `
  --output-dir artifacts\runs\vdn_syncg\seed_20260720

# 训练完成后先做签名/逐轮/检查点审计，再依次跑 clean、五种退化和 RPM，
# 最后生成配对分组 bootstrap 表。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_vdn_evaluations.ps1

# 再完整训练、验证并评测另外两个独立种子。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_vdn_baseline_replicates.ps1

# 核对三次训练和 21 组逐样本评测，并生成当前论文主表。
python -m experiments.summarize_vdn_replicates `
  --bootstrap-iterations 5000 `
  --output artifacts\runs\vdn_syncg\vdn_replicate_summary.json
```

VDN 只替换指针方向估计；表盘检测、起终点、已知量程换算、失败惩罚和退化图像均与主方法
共享。论文中必须写作 `VDN architecture, retrained on SyncG`，不能冒充官方预训练端到端
结果。评估同时输出 VDN 原生方向 MAE 和适配后的 NMAE/Acc@2%。正式验证器会重新构造
652/73 个训练/验证场景组，核对零交集、样本 ID 哈希、学习率、损失权重、优化器步数、源码
与权重哈希；共享缓存还必须具有完全相同的退化协议、退化源码和检测器权重签名。

三随机种子只重复新增残差/门控拟合，复用同一份冻结图像预测，避免无意义地重复 24,000 次
视觉推理：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_seed_stability.ps1
```

输出为 `artifacts/runs/seed_stability/seed_stability.{json,md}`；正式三种子固定为
`20260720/20260721/20260722`。汇总器逐条件核对预测 JSONL 的 SHA-256 和完整前端签名，
并核对每份指标实际引用的校准器，保证变化只来自残差/门控拟合种子。

正式结果位于三个 `artifacts/runs/vdn_syncg/seed_*` 目录及
`vdn_replicate_summary.{json,md}`。三次 VDN 的验证方向 MAE 为
`0.6743°±0.0300°`。原掩码分支相对首个 VDN 诊断种子在 clean、两档模糊和中度透视上
取得显著更低的 NMAE；
重度透视统计持平，组合重度退化和 RPM 则由 VDN 显著领先。RPM 差距主要来自原分支
72.73% 对 VDN 98.78% 的端到端 coverage；这一冻结诊断驱动了第 9 节的分割无关方向回退，
而没有使用外测标签继续调残差或门控阈值。

## 9. 第一代独立支点—方向头与双表示路由（历史消融）

RPM 失败归因显示，原掩码分支的主要问题是指针分割/中心线校验无输出。为此新增一个与
分割无关的 torchvision ResNet-18：同一编码器同时预测 `64×64` 支点热图与全局二维单位
方向。该实现不导入或复制 VDN 的 GPL 源码；VDN 仍是独立外部基线。

正式训练只使用 SyncG train，并按 `meter type + scene` 分组留出验证集：

```powershell
python -m experiments.train_pivot_direction_syncg `
  --output-dir artifacts\runs\pivot_direction_syncg\seed_20260722 `
  --epochs 30 --batch-size 48 --workers 4 --seed 20260722

python -m experiments.verify_pivot_direction_run `
  --run-dir artifacts\runs\pivot_direction_syncg\seed_20260722 `
  --expected-epochs 30 --expected-batch-size 48 --expected-seed 20260722

# 对 clean、五种控制退化和 RPM 做冻结评测，并生成硬失败路由的配对统计。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_pivot_direction_experiments.ps1
```

正式硬路由没有可调阈值：原掩码/残差方法有输出时保留原值，只有无输出才调用方向头；两者
都失败才保留失败。RPM 标签不用于训练、模型选择或路由。评估器复用 VDN 评测中已冻结的
表盘框与起终参考，只读取这些共享前端几何，不读取 VDN 的方向或读数。

主种子最佳验证方向 MAE 为 1.6873°，支点误差为输入宽度的 0.00559。RPM 上方向头恢复
468/490 次原分支失败，最终 coverage 从 0.7273 提高到 0.9878，NMAE 从 0.4973 降到
0.3346；相对 VDN 的 0.3813，配对 `ΔNMAE=-0.0467`，95% CI 为
`[-0.0889, -0.0199]`。但最终 Acc@2% 为 0.0484，低于 VDN 的 0.0595；严重组合退化的
NMAE 0.2510 也仍差于 VDN 0.2364。论文不得省略这两个边界。

三次完整训练只重复方向头，clean/RPM 共享同一冻结前端：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_pivot_direction_replicates.ps1

python -m experiments.summarize_pivot_direction_replicates
```

汇总器核对每个检查点、训练验证器、评测输出、源码签名、共享前端哈希和逐样本 ID，输出
`artifacts/runs/pivot_direction_syncg/replicate_stability.{json,md}`。主种子的七条件结果位于
`seed_20260722/dual_route_comparison.{json,md}`。

正式三种子结果：验证方向 MAE `1.7099 ± 0.1726°`；RPM 方向分支 NMAE
`0.3439 ± 0.0029`，Ours-hard 双表示 NMAE `0.3317 ± 0.0039`、Acc@2%
`0.0482 ± 0.0003`、coverage `0.9878 ± 0.0000`。三个种子都恢复 468/490 次原分支
硬失败；双表示相对固定 VDN 的 NMAE 差为 `-0.0496 ± 0.0039`。

## 10. 第一代训练侧跨模型 OOF 质量路由（历史消融）

硬回退只解决“无输出”，不能处理掩码分支已有有限读数但误差很大的情况。质量路由训练集由
三个方向头验证集并集构成：每个样本的方向预测来自从未训练过该仪表组的检查点，掩码读数
来自原残差模型的 grouped-OOF 预测。最终得到 4,380 张、197 组，组泄漏和测试样本使用均为
0。RPM 和 SyncG test 不支持作为本脚本的训练输入。

```powershell
# 约 2 分钟：重跑检测和三个 held-out 方向检查点，生成跨模型 OOF 配对。
python -m experiments.collect_quality_router_oof --overwrite

# 约数秒：五折 grouped OOF 拟合 ExtraTrees 收益回归器并冻结阈值。
python -m experiments.train_quality_router --overwrite

# 只用上述训练侧 OOF 做特征消融。
python -m experiments.ablate_quality_router_features --overwrite
```

路由第一层保持硬失败规则不变；共同成功时，才在预测收益大于训练侧阈值 `0.0056015` 时切换
到方向分支。运行时特征包括分支读数/角度分歧、支点热图峰值、掩码质量、残差不确定性与
表盘检测置信度，不读取真值、样本 ID、数据集名或环境标签。

七组冻结评测可分别调用 `experiments.evaluate_quality_router`；已有全部方向/VDN 缓存时，
以下入口会连同三个方向种子的 clean/RPM 稳定性、汇总和审计一起执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_quality_router_experiments.ps1

# 若训练侧 OOF 与路由模型已冻结，只重做评测：
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_quality_router_experiments.ps1 -SkipTraining
```

正式主种子 NMAE：clean `0.1071`、severe blur `0.1260`、severe perspective `0.1622`、
combined severe `0.2316`、RPM `0.3241`。六个 SyncG 条件相对硬回退的配对区间均低于 0；
RPM 相对硬回退只有点估计改善、区间跨 0，但相对 VDN 的 NMAE 仍显著更低。三个方向种子的
clean/RPM 最终 NMAE 为 `0.1075 ± 0.0005` / `0.3238 ± 0.0037`。

完整结果位于 `artifacts/runs/quality_router_syncg/quality_router_comparison.{json,md}`，训练
消融位于 `feature_ablation.{json,md}`，`verification.json` 逐行重算七组路由和指标。方法构想
受此前 test 失败分析启发，所以当前复评不冒充全流程盲测；任何后续修改都必须另设未使用
测试集。

## 11. 概率圆周方向、透视等变训练与不确定性融合（中间消融）

在 v1 独立方向头和质量路由之上，第二阶段模型同时预测支点热图、直接二维方向、72-bin
圆周分布与角方差。训练视图包含精确单应变换后的支点/射线标签，并对两个视图的预测增加
可微等变一致性。最终方向由直接向量和圆周 resultant vector 联合解码。

```powershell
# 单个或三个完整种子；默认 30 epochs、batch 24。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_probabilistic_direction_training.ps1

# 主种子七条件冻结评测。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_probabilistic_direction_evaluations.ps1

# 同权重 direct/circular/fused 解码消融，以及重新训练的等变/投影消融。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_probabilistic_decoder_ablations.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_probabilistic_direction_ablations.ps1
```

这一阶段还评估了根据两个专家条件对数误差方差进行逆方差加权的软融合。mask/vector
任一失败时仍保留确定性硬失败规则。训练数据仍只有 SyncG train：三个方向模型验证集的并集
提供 group-held-out vector 预测，mask 使用原 grouped-OOF 预测；融合器内部再做五折
GroupKFold。SyncG test、控制退化与 RPM 均不进入拟合。

```powershell
# 跨模型 OOF、融合拟合、原生不确定性特征消融、七条件评测、逐行复核。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_uncertainty_fusion_experiments.ps1
```

完整算法、损失、无泄漏协议与创新性边界见
[`../docs/PROBABILISTIC_FUSION_METHOD_CN.md`](../docs/PROBABILISTIC_FUSION_METHOD_CN.md)。方向 MAE
与端到端 NMAE 必须分开报告：强透视下像平面方向更准，不等于角度到刻度进度的投影非线性
已经被完全消除。软融合是负面/中间消融，不是当前最终方法。

## 12. 透视感知进度校准与最终选择路由

概率方向模型显著降低了方向角 MAE，但冻结参考点的 `start_angle/range_angle` 仍限制最终读数。
新增校准器只在 4,380 条 SyncG/train grouped-OOF 行上拟合归一化 progress residual，并输出
ExtraTrees 树间标准差。第二层路由在 mask 与 calibrated vector 之间选择，硬失败回退优先级
保持不变；SyncG test、控制退化和 RPM 均不进入拟合或阈值选择。

```powershell
# 已有概率 OOF 与冻结七条件预测时，约数分钟完成训练、评测、汇总和逐行审计。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_calibrated_progress_experiments.ps1

# 只重做冻结评测和复核。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_calibrated_progress_experiments.ps1 -SkipTraining
```

训练侧 nested-OOF：raw vector / calibrated vector / final router NMAE 分别为
`0.14872 / 0.09651 / 0.08985`，最终 Acc@2% 为 `0.46370`。冻结最终路由结果：clean
`0.08901`、severe blur `0.10815`、severe perspective `0.14539`、combined severe
`0.21837`、RPM `0.28420`。六个 SyncG 条件相对旧质量路由的 95% 区间均低于 0；RPM
只有六个组，区间跨 0。

同协议外部对比使用公开 VDN 架构在相同 SyncG train 上重训，并共享表盘框、起终参考、
已知量程适配、退化输入和全分母失败计分。最终方法在七个条件的 NMAE 均更低，配对分组
bootstrap 区间也均低于 0；但 severe perspective 和 RPM 的 Acc@2% 仍低于 VDN，不能写成
所有指标全面领先。自动汇总中的第二张表专门给出 VDN 与最终方法，避免把内部消融当作外部
模型对比。

自动生成：

- `artifacts/runs/calibrated_progress_router_syncg/calibrated_progress_comparison.{json,md}`；
- `artifacts/runs/calibrated_progress_router_syncg/verification.json`；
- 校准器与路由器各自的训练 summary、grouped-OOF 诊断和七条件逐样本预测。

算法定义、边界和方向/进度消融见
[`../docs/PROBABILISTIC_FUSION_METHOD_CN.md`](../docs/PROBABILISTIC_FUSION_METHOD_CN.md)，阶段总报告见
[`../docs/MODEL_PROGRESS_REPORT_CN_20260723.txt`](../docs/MODEL_PROGRESS_REPORT_CN_20260723.txt)。

## 13. 参考分支感知的安全校准（新测试集前冻结）

旧进度校准器对所有参考点状态使用同一个残差回归器。训练 OOF 误差分层显示：
`start_and_end` 分支的原始 vector 已有很高的精细精度，而 `start_only`、`end_only` 和
默认参考范围仍需要较大校正。全局回归会在修复大误差的同时扰动一部分本来准确的双参考点
样本。

新候选方法因此增加两层约束：

1. 按 `default_start_end / start_only / end_only / start_and_end` 分别拟合残差回归器；
2. 每个分支分别选择校正幅度上限和安全死区。只有预测残差绝对值越过死区才执行校正，
   否则保留原始 vector progress。

外层每个仪表组的预测只来自未见过该组的模型；校正幅度、死区和路由阈值均在该外层训练
部分内部再做 5 折 GroupKFold 选择。SyncG test、控制退化、RPM 和后续现场新测试集均未
参与本轮拟合或选型。树预测按固定 estimator 顺序聚合，避免并行浮点求和在路由阈值附近
产生跨机器末位漂移；固定种子的重复运行会生成相同的逐样本 OOF 文件。

```powershell
# 复现校准器和最终路由的 5×5 嵌套训练侧 OOF；约 2 分钟。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\run_reference_conditioned_training.ps1
```

同一外层折和同一全分母计分下的训练侧消融：

| 方法 | NMAE | Acc@2% |
|---|---:|---:|
| Raw probabilistic vector | 0.148718 | 0.30571 |
| 旧全局校准器 v1 | 0.096511 | 0.37900 |
| 严格嵌套全局残差 | 0.094198 | 0.39018 |
| 分支残差、无安全死区 | 0.093348 | 0.40685 |
| 分支残差、共享死区 | 0.092946 | 0.41553 |
| 分支残差、分支死区（完整） | **0.091401** | **0.44680** |
| mask/完整校准 vector 最终路由 | **0.088271** | **0.48767** |

完整校准相对严格嵌套全局残差的 NMAE 差为 `-0.002797`，197 组 bootstrap 95% CI
`[-0.003579, -0.002020]`；相对“分支残差、无死区”为 `-0.001947`
`[-0.002347, -0.001528]`。最终路由相对旧全局校准最终路由的差为 `-0.001579`
`[-0.002303, -0.000850]`，Acc@2% 从 `0.46370` 提高到 `0.48767`。

双参考点分支只对约 8.5% 的成功样本执行校正；其 NMAE 从原始 `0.06420` 降到
`0.04830`，同时 Acc@2% 保持为 `0.86411`，而不是像旧全局校准那样大范围扰动高精度
样本。这里的 ExtraTrees 不是创新点；可写的算法贡献是“参考几何状态条件化 +
训练侧选择的安全拒绝校正 + 嵌套分组协议”。

本轮不运行已经多次查看过的 SyncG test 或 RPM，以免继续按旧测试反馈调参。新现场测试集
准备好后，只执行一次冻结入口：

```powershell
python -m experiments.evaluate_reference_conditioned_pipeline `
  --raw-predictions <raw.jsonl> `
  --base-predictions <base.jsonl> `
  --vector-predictions <vector.jsonl> `
  --reference-predictions <vdn_and_reference.jsonl> `
  --output-dir artifacts\runs\reference_conditioned_router_syncg\evaluations\field_holdout `
  --condition field_holdout `
  --overwrite
```

评估器会校验校准器、路由器、训练来源和源码 SHA-256，并输出逐样本分支、死区是否触发、
路由结果、全分母 NMAE/Acc@1%/2%/5%、coverage 和分组 bootstrap。训练产物位于
`artifacts/runs/reference_conditioned_progress_calibrator_syncg/model/` 与
`artifacts/runs/reference_conditioned_router_syncg/model/`；训练入口最后还会逐行重算校正、
路由和指标，写入 `artifacts/runs/reference_conditioned_router_syncg/verification.json`。

## 14. 接回服务

服务启动前指定同一个分割检查点和推理设备；不设置时仍使用仓库自带权重和 CPU：

```powershell
$env:POINTER_METER_SEGMENTATION_WEIGHTS = `
  (Resolve-Path artifacts\runs\syncg_segmentation\best.pt).Path
$env:POINTER_METER_DEVICE = "cuda"
python main.py
```

请求中再配置选择性残差校正器：

```json
{
  "reading_backend": "geometry_fusion_weighted_calibrated",
  "residual_calibrator_path": "artifacts/runs/syncg_full/calibrator.joblib",
  "correction_mode": "off",
  "auto_zero": false
}
```

若模型缺失、门控拒绝或不确定性过高，服务返回质量加权几何基线；不会把外部测试标签带入运行时。

## 15. Pointer-10K 零样本方向部件外测

Pointer-10K 不进入标量读数主表。固定协议只解压官方 test，验证外层 ZIP、标注文件、539 个
test 图像和 685 个指针实例的身份；再按标注数量预先保留 438 张单指针图像，排除 101 张
多指针图像。筛选规则不读取模型预测，Pointer-10K train/validation 不解压、不训练、
不校准。官方表盘框用于隔离“方向估计”部件能力，因此结果不能表述为端到端检测成绩。

```powershell
python -m experiments.extract_pointer10k_test `
  --archive D:\BaiduNetdiskDownload\Pointer-10K.zip `
  --output datasets\Pointer10K_official\pointer_10k

python -m experiments.datasets pointer10k `
  --root datasets\Pointer10K_official\pointer_10k `
  --output artifacts\manifests\pointer10k_single_pointer_test.jsonl

python -m experiments.evaluate_pointer10k_direction `
  --manifest artifacts\manifests\pointer10k_single_pointer_test.jsonl `
  --checkpoint artifacts\runs\vdn_syncg\seed_20260720\best.pt `
  --model-kind vdn --model-label VDN `
  --output artifacts\runs\pointer10k\formal\vdn.jsonl `
  --vdn-source datasets\Pointer10K_official\reference-vdn `
  --device cuda --batch-size 64 --workers 4

foreach ($seed in 20260721,20260722) {
  python -m experiments.evaluate_pointer10k_direction `
    --manifest artifacts\manifests\pointer10k_single_pointer_test.jsonl `
    --checkpoint "artifacts\runs\vdn_syncg\seed_$seed\best.pt" `
    --model-kind vdn --model-label "VDN-seed$seed" `
    --output "artifacts\runs\pointer10k\formal\vdn_seed$seed.jsonl" `
    --vdn-source datasets\Pointer10K_official\reference-vdn `
    --device cuda --batch-size 64 --workers 4
}

python -m experiments.evaluate_pointer10k_direction `
  --manifest artifacts\manifests\pointer10k_single_pointer_test.jsonl `
  --checkpoint artifacts\runs\probabilistic_pivot_direction_syncg\seed_20260722\best.pt `
  --model-kind probabilistic --model-label Ours `
  --output artifacts\runs\pointer10k\formal\ours_seed20260722.jsonl `
  --device cuda --batch-size 64 --workers 4

# HARR 官方发布权重：不同训练来源，只作跨协议 pointer-branch 补充对比。
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\experiments\setup_harr_baseline.ps1 `
  -Proxy http://127.0.0.1:7890

python -m experiments.evaluate_pointer10k_direction `
  --manifest artifacts\manifests\pointer10k_single_pointer_test.jsonl `
  --checkpoint artifacts\vendor\Detect-and-read-meters\model\meter_data\textgraph_vgg_100.pth `
  --model-kind harr --model-label HARR-official-v2 `
  --output artifacts\runs\pointer10k\formal\harr_official.jsonl `
  --harr-source artifacts\vendor\Detect-and-read-meters `
  --device cpu --batch-size 4 --workers 0

python -m experiments.summarize_pointer10k_direction `
  --method "VDN=artifacts\runs\pointer10k\formal\vdn.jsonl,artifacts\runs\pointer10k\formal\vdn_seed20260721.jsonl,artifacts\runs\pointer10k\formal\vdn_seed20260722.jsonl" `
  --method "HARR-official-v2=artifacts\runs\pointer10k\formal\harr_official.jsonl" `
  --method "Ours=artifacts\runs\pointer10k\formal\ours_seed20260720.jsonl,artifacts\runs\pointer10k\formal\ours_seed20260721.jsonl,artifacts\runs\pointer10k\formal\ours_seed20260722.jsonl" `
  --baseline-label VDN `
  --output-json artifacts\runs\pointer10k\pointer10k_extended_comparison.json `
  --output-md artifacts\runs\pointer10k\pointer10k_extended_comparison.md
```

`summarize_pointer10k_direction` 支持把三个独立训练种子写成同一个
`LABEL=run1,run2,run3`，同时输出逐图配对 bootstrap 和全方法两两比较。
角度 MAE 是针尾到针尖方向的最小圆周夹角均值；`Acc@5°`/`Acc@10°` 是误差不超过相应
角度的全体样本比例；失败按 180° 计入 MAE 且在准确率中计错，coverage 另报。自然低质量
组由模糊、低照度、低对比、小表盘四个预测无关统计的底四分位中至少命中两项构成，它不是
Pointer-10K 官方 LQPI 标签。完整结果与论文表述边界见
[`../docs/POINTER10K_RESULTS_CN.md`](../docs/POINTER10K_RESULTS_CN.md)。

## 16. 统一复杂度、batch=1 延迟与失败分解

同一设备上的方法比较使用活动模型栈基准：一次表盘检测、当前实现中的两次参考点检测，以及
各方法实际需要的读数模块。FLOPs 采用实际输入分辨率、一次乘加计两次操作；Ours 的总时间
还包含两个冻结 ExtraTrees 的 CPU 单样本预测：

```powershell
python -m experiments.benchmark_model_complexity `
  --device cuda --precision amp_fp16 `
  --warmup 30 --iterations 100 --tree-n-jobs 1 `
  --output artifacts\runs\efficiency\model_stack_efficiency.json
```

RTX 4060 正式结果：

| 方法 | 参数量 M | FLOPs G | 峰值显存 MiB | 神经栈 ms | CPU 后处理 ms | 总均值/中位/P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Original Transformer | 138.47 | 110.44 | 700.99 | 42.51 | 0.00 | 43.67 / 41.24 / 51.78 |
| VDN | 34.23 | 81.66 | 198.63 | 20.66 | 0.00 | 20.81 / 20.38 / 23.59 |
| Ours-final | 34.07 | 95.10 | 304.78 | 34.36 | 61.61 | 99.55 / 97.18 / 101.75 |

该统一表排除图像解码、resize/normalize、YOLO NMS、裁剪、渲染和 JSON；用于比较活动模型
复杂度，不替代下述真实逐图缓存的端到端时间。Ours 的 CPU 后处理含两个各 600 棵树的模型，
batch=1 固定为单线程以避免并行调度改变硬件口径，预测值不变。

正式预测缓存由 `collect_predictions` 逐图串行生成，因此每行的 `runtime_seconds` 是
batch=1 墙钟时间：从图像读取开始，到完整读数 payload 生成结束；不含模型初始化和 JSON
写盘。以下命令不重复推理，只对冻结缓存汇总全部样本，失败和提前退出不从分母删除：

```powershell
python -m experiments.make_latency_table `
  --dataset "SyncG=artifacts\predictions\syncg_test.jsonl" `
  --dataset "RPM-10K single-pointer=artifacts\predictions\rpm10k_single_pointer_test.jsonl" `
  --output artifacts\runs\latency_table.md

python -m experiments.make_failure_table `
  --dataset "SyncG=artifacts\predictions\syncg_test.jsonl" `
  --dataset "RPM-10K single-pointer=artifacts\predictions\rpm10k_single_pointer_test.jsonl" `
  --output artifacts\runs\failure_table.md
```

当前 RTX 4060 冻结缓存的全部样本结果：SyncG 平均/中位/P95 为
`129.74/128.15/163.99 ms`（串行 `7.71 FPS`）；RPM-10K single-pointer 为
`179.45/133.05/592.64 ms`（`5.57 FPS`）。RPM 长尾包含高分辨率真实图像和完整成功路径；
未检出表盘或指针的样本会提前退出，延迟更短，所以不能只报告成功子集或只报告失败子集。

## 实验纪律

- SyncG 官方 train/test 不互换；OOF 分组默认来自场景与表型组合。
- RPM-10K 从第一次运行开始就使用冻结的特征、模型、残差截断、不确定性阈值和门控阈值；不使用其训练集。
- Pointer-10K 仅作官方 test 单指针方向外测，train/validation 使用量为 0，不能回看其结果选择模型或阈值。
- 所有方法使用同一个预测缓存，失败样本计入 coverage，不能只在成功子集上宣称更优。
- 主表报告 NMAE 与 coverage；MAE 只在量程一致的子集内有直接可比性。
- 公开数据与实验产物均被 Git 忽略；脚本不会生成或伪造论文数值，只有完整下载和冻结评测完成后才写主表。

算法组件的快速单测：

```powershell
python -m unittest discover -s test -p "test_*.py"
```
