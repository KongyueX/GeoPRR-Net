# 务实的论文实验方案

2026-07-21 的主实验与六组控制退化正式运行已经结束。聚合结果、置信区间与可安全主张的
结论见 [`../docs/FORMAL_RESULTS_CN.md`](../docs/FORMAL_RESULTS_CN.md)；
本文件继续作为复现实验协议和命令说明。

本目录只使用两个正式数据集：

- **SyncG**：只用官方 `train` 微调指针分割、训练选择性残差校正器，并使用官方 `test` 做同域测试。
- **RPM-10K**：只做冻结模型的零微调真实图像外部测试，不用于选特征、阈值或超参数。

仓库原有的 30+8 张图只适合作为补充案例，不并入主表，也不作为“独立数据集”反复调参。这样既避免五套数据集雨露均沾，也能把核心论点讲清楚：质量加权融合是否有效，以及学习残差在跨域时能否通过门控避免负迁移。

官方下载入口：[SyncG 代码与数据说明](https://github.com/SKYWOWDYH/SyncG)、[SyncG 数据](https://huggingface.co/datasets/YihengDeng/syncG)、[RPM-10K / DialBench](https://github.com/Event-AHU/DialBench)。固定提交中的 SyncG 数据卡标注为 CC BY 4.0，使用及论文中应保留数据集归属；RPM-10K 仓库当前把数据许可标为 TBD，投稿或再分发前必须再次核对。本仓库只提供下载脚本，不分发数据。

外部同类模型不能由内部变体替代；VDN 等方法的可复现性和公平接入规则见
[`../docs/BASELINE_AUDIT_CN.md`](../docs/BASELINE_AUDIT_CN.md)。

## 七个内部方法（消融表）

所有读数方法共享同一次前端推理缓存。分割网络只允许在 SyncG `train` 内微调；进入读数实验后，检测、校正、分割和起终点网络全部冻结。

| 名称 | 含义 |
|---|---|
| Original Transformer | 项目原始的分割掩码 + meter transformer 读数基线 |
| Geometry-v1 | 直线拟合后取最远针尖 |
| Geometry-v2 | 外侧候选点稳健方向投票 |
| Mean Fusion | v1/v2 简单平均，保留为消融 |
| Quality-weighted Fusion | 使用轴线一致性、针尖支持、方向投票集中度与正反侧分离度估计质量并加权 |
| Residual without Gate | 在质量加权结果上无条件加残差 |
| Ours | 质量加权 + 归一化残差 + 不确定性/学习门控 |

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

## 8. 接回服务

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

## 实验纪律

- SyncG 官方 train/test 不互换；OOF 分组默认来自场景与表型组合。
- RPM-10K 从第一次运行开始就使用冻结的特征、模型、残差截断、不确定性阈值和门控阈值；不使用其训练集。
- 所有方法使用同一个预测缓存，失败样本计入 coverage，不能只在成功子集上宣称更优。
- 主表报告 NMAE 与 coverage；MAE 只在量程一致的子集内有直接可比性。
- 公开数据与实验产物均被 Git 忽略；脚本不会生成或伪造论文数值，只有完整下载和冻结评测完成后才写主表。

算法组件的快速单测：

```powershell
python -m unittest test.test_selective_geometry
```
