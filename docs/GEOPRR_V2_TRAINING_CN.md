# 轻量几何读数模型的训练与评估

## 环境与目录

目标环境为 Ubuntu 22.04、Python 3.10、RTX PRO 6000 Blackwell 96GB。独立环境使用 PyTorch 2.11.0 / torchvision 0.26.0 的 CUDA 12.8 构建及 timm 1.0.30；基础镜像的 CUDA 11.8 与其他项目环境保留。标准 PyTorch 算子不需要编译自定义 CUDA 扩展。

| 用途 | 路径 |
|---|---|
| 项目目录 | `/root/autodl-tmp/geoprrnet/` |
| 代码 | `/root/autodl-tmp/geoprrnet/code/` |
| 独立 Python | `/root/autodl-tmp/geoprrnet/.venv/bin/python` |
| 数据 | `/root/autodl-tmp/geoprrnet/datasets/` |
| 新实验输出 | `/root/autodl-tmp/geoprrnet/runs/geoprr_v2/` |
| 预训练缓存 | `/root/autodl-tmp/geoprrnet/cache/` |

首次安装：

```bash
cd /root/autodl-tmp/geoprrnet/code
bash experiments/setup_geoprr_v2.sh
```

每次登录后激活：

```bash
cd /root/autodl-tmp/geoprrnet/code
source experiments/geoprr_v2_env.sh
```

依赖文件为 `experiments/requirements-geoprr-v2.txt`。环境激活同时将预训练缓存置于数据盘，并设置 CPU 数学库线程数。服务器已缓存两个公开检查点，使用 `HF_HUB_OFFLINE=1` 加载。迁移到可访问 Hugging Face 的新机器时，先执行 `export HF_HUB_OFFLINE=0`，再用下方检查命令下载缓存。无卡模式可执行 CPU 检查；GPU 启用后执行 CUDA 检查：

```bash
python -m experiments.check_geoprr_v2_runtime \
  --device cuda --pretrained --image-size 768 --batch-size 2 \
  --output "$GEOPRR_HOME/runs/geoprr_v2/setup/cuda_check.json"
```

该命令检查两种骨干的完整模型前后向、概率归一化和梯度有限性。输出中的单步时间用于环境诊断，不作为正式推理延迟。

## 模型与初始配置

`GeoPRRV2` 推理仅接受 ImageNet 归一化的 RGB ROI。共享骨干的多尺度特征用于预测刻度进度场 U、表盘平面虚拟读出点 Q 和几何有效域 M。仅在 `sum(Q*M)` 达到阈值时归一化场读数，其余样本使用同一编码器的外观头回退；最终评估包含全部样本。

| 变体 | 结构监督 | 最终读数 |
|---|---|---|
| `raw_scalar` | 无 | 全局标量回归 |
| `raw_distribution` | 无 | 全局有序分布均值 |
| `multitask` | U、Q、M | 全局分布，检验额外监督的作用 |
| `field` | U、Q、M | 场组合与低支持回退 |

两种骨干均支持以上变体及端到端微调：

- `convnextv2_femto.fcmae_ft_in1k`，默认主骨干。
- `mobilenetv4_conv_medium.e500_r224_in1k`，另一种轻量骨干。

两个配置统一采用 768×768 输入、192×192 结构输出、256 个有序读数格、64 通道解码器、batch 32、8 个数据加载进程。初值为 5 轮骨干冻结热启动及 60 轮联合优化，AdamW，骨干/新增层学习率分别为 1e-4 / 5e-4，余弦衰减，AMP 与梯度裁剪。它们是初始训练设置，尚不能视为最佳超参数；正式 batch 与吞吐需在启用 GPU 后实测。

支持阈值初值为 0.15。阈值和几何标签参数随运行记录；后续只在源域开发集选择并冻结，外部测试不调阈值。

## 完整数据准备

SyncG 根目录应包含 `annotations/train`、`annotations/test`、`images/train`、`images/test`。上传完成后运行：

```bash
python -m experiments.prepare_geoprr_v2_data \
  --root "$GEOPRR_HOME/datasets/SyncG/syncG" \
  --output-dir "$GEOPRR_HOME/runs/geoprr_v2/manifests"
```

准备程序核对既有官方发布名单和数量（训练 16,000，测试 4,000），并要求每份标注的对应图片已经存在。官方训练集内部按全局场景划分训练和验证，输出 `train.jsonl`、`validation.jsonl`、`source_train.jsonl`、`test.jsonl` 与固定的 `split.json`。官方测试集不参与训练或选模。可用 `--split PATH` 复用已经确定的场景划分。

不完整上传仅用于诊断时，可传 `--diagnostic-limit N`，输出名称带 `diagnostic_`；该名单不能作为完整训练名单或正式性能证据。

## 标签合同审计

正式训练前检查全部源训练标注：

```bash
python -m experiments.audit_geoprr_v2_geometry \
  --manifest "$GEOPRR_HOME/runs/geoprr_v2/manifests/source_train.jsonl" \
  --dataset-root "$GEOPRR_HOME/datasets" \
  --image-size 768 --min-support-mass 0.15 \
  --output "$GEOPRR_HOME/runs/geoprr_v2/geometry_audit.json"
```

H 的输入是 bbox 归一化相机坐标；ROI 裁剪、缩放与输出网格采用同一半像素中心约定。在 H 整流后的平面上，以标注的主刻度角间隔约束四参数相似拟合，估计规范中心、参考半径和方向；观测刻度的径向与角度残差仍按原阈值核验。Q 从物理读数及该刻度环投影得到，可见针尖不会直接充当 Q 标签。

审计报告包含刻度圈残差、刻度顺序和角度—读数一致性、GT Q/M/U 组合误差及有效质量。失败行保留，命令以非零状态结束；不通过静默过滤失败行生成更容易的训练集。`--probe-endpoints` 可额外检查每个布局的端点数值表示，报告会将其标为标签探针。

Q 的核密度在参考半径为 1 的共同规范平面上定义，标准差为 `2 * point_sigma / output_size`，再乘投影面积 Jacobian 并归一化为输出像素质量。这个定义与跨相机 Q 的质量传输使用同一物理表面。

## 训练与恢复

数据上传完整且 GPU 已启用后，可用下列入口执行数据准备、全量源域标签审计，并依次训练 Femto 与 MobileNetV4：

```bash
bash experiments/run_geoprr_v2_training.sh
```

该入口只启动两种骨干的源域场模型训练。以下单模型命令用于分别启动或恢复。

Femto 主模型：

```bash
python -m experiments.train_geoprr_v2 \
  --config experiments/configs/geoprr_v2_femto.json \
  --train-manifest "$GEOPRR_HOME/runs/geoprr_v2/manifests/train.jsonl" \
  --val-manifest "$GEOPRR_HOME/runs/geoprr_v2/manifests/validation.jsonl"
```

MobileNetV4 主模型：

```bash
python -m experiments.train_geoprr_v2 \
  --config experiments/configs/geoprr_v2_mobilenetv4.json \
  --train-manifest "$GEOPRR_HOME/runs/geoprr_v2/manifests/train.jsonl" \
  --val-manifest "$GEOPRR_HOME/runs/geoprr_v2/manifests/validation.jsonl"
```

对照用同一命令追加 `--variant raw_scalar`、`--variant raw_distribution` 或 `--variant multitask`；两种骨干和各变体使用独立输出目录。重复种子可通过 `--seed` 指定，骨干对照使用同一组种子。

默认输出结构为 `runs/geoprr_v2/{config.name}/{variant}/seed_{seed}/`，包含 `run.json`、`config.json`、`history.jsonl`、`latest.pt`、`best.pt` 和 `best_validation.json`。最优检查点只按源域验证 NMAE 选择。

恢复时重用原命令及其参数，并追加：

```bash
--resume "$GEOPRR_HOME/runs/geoprr_v2/convnextv2_femto/field/seed_20261019/latest.pt"
```

恢复包含模型、优化器、调度器、AMP scaler、随机状态和 epoch；每轮用 `seed + epoch` 确定数据加载顺序。配置与名单须和原运行相同。有限 batch 的 smoke 检查点只用于诊断，不继续作为正式训练。

## 冻结评估

例如评估独立官方 SyncG 测试集：

```bash
python -m experiments.evaluate_geoprr_v2 \
  --checkpoint "$GEOPRR_HOME/runs/geoprr_v2/convnextv2_femto/field/seed_20261019/best.pt" \
  --manifest "$GEOPRR_HOME/runs/geoprr_v2/manifests/test.jsonl" \
  --output "$GEOPRR_HOME/runs/geoprr_v2/evaluation/femto_syncg_test"
```

其他 ROI 数据集使用相同的冻结评估入口，manifest 需提供 `sample_id`、`group_id`、图像路径以及真实读数/量程或 `normalized_progress`。已裁剪图像用 `roi_path`；全图用 `image_path` 和 `metadata.dial_bbox`。评估不读取结构标注。更换主机路径后，应将图像路径指向实际文件，避免把缺失路径当成模型失效。

输出 `predictions.csv` 和 `summary.json`，报告 NMAE、Acc@1/2/5%、覆盖率、分位误差、尾部误差及数据集/条件/组汇总。图像或 ROI 失败保留真实目标，误差按 1.0 计入完整分母。计时明确区分网络前向和包含读图、裁剪、归一化、传输、解码及结果收集的批量流程。

## 受控配对训练

`--pair-manifest PATH` 接受逐行 JSON 配对记录：`sample_a`、`sample_b`、`layout_id`、`kind`（`state` / `appearance` / `camera`）、`factors_a`、`factors_b`。每份 factors 必须包含实际 `g`、`r`、`a`、`c` 参数，并与对应图像 `metadata.factors`、`metadata.layout_id` 一致。指定干预因素必须变化，其余因素必须保持；物理状态 r 必须等于真实归一化读数。

配对双方须处于同一训练划分。跨图组合通过共同表盘平面反采样，GT 支持决定配对资格，低预测支持保留质量惩罚。直接回归与多任务对照同样读取配对 RGB，避免将额外图像暴露混同于组合机制的收益。

## 验证与已知限制

必要测试：

```bash
python -m unittest discover -s test -p 'test_geoprr_v2_*.py' -v
```

数值检查和短步训练只证明实现可运行。全量源域合同、CUDA 显存/吞吐与最终精度需在数据上传及 GPU 启用后验证。高分辨率减小离散标签误差，但不能保证所有极端投影都通过合同；全量失败需检查标注、裁剪或采样定义。

原始发布数据未提供完整受控因素记录时，默认训练不启用受控配对；相关接口需新渲染数据及其实际因素记录。普通光度增强不作为因素干预证据。当前实现没有新模型的正式性能结果，历史 GeoPRR 结果不能填入本方法。
