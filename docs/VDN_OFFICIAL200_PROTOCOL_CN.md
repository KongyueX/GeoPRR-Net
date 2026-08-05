# VDN official-200 三种子统一终止协议

## 1. 决策背景

VDN 100→150 epochs 的 Phase-2 训练制品已经完成独立复核。三个种子的训练
制品本身均完整，但 seed `20260721` 的末两段 10-epoch 验证角度 MAE 均值绝对
相对变化为 `1.6081%`，超过预声明的严格 `<1%` 收敛门。因此 Phase-2 三种子
cohort 失败封闭，不授权任何 supporting test、public、field、sealed 或
confirmatory 评估。

下一阶段不是只延长失败 seed，也不是继续接续 150-epoch 权重，而是对三个正式
seed 采用完全相同的官方 200-epoch 预算，从官方 ImageNet ResNet-18 初始化重新
训练。这样避免了事后按 seed 追加预算带来的选择偏差。

## 2. 冻结科学配置

| 项目 | 冻结值 |
|---|---:|
| seeds | `20260720`, `20260721`, `20260722` |
| 初始化 | pinned VDN + 官方 ResNet-18 ImageNet 权重；每个 seed 从头初始化 |
| 数据范围 | 仅 SyncG official train manifest；按 seed 做 grouped 90/10 train/validation |
| epochs | 每个 seed 恰好 `200` |
| batch size | `8` |
| workers | `4`，每个 epoch 重建 DataLoader，`persistent_workers=False` |
| image size | `384×384` |
| optimizer | 完整冻结参数组 schema 的 Adam，weight decay `0` |
| AMP | 启用；初始 scale `512`；逐 batch 记录 skip/replay 审计 |
| vector loss 权重 | `(epoch - 1) / 199`，epoch 1 为 `0`，epoch 200 为 `1` |
| LR epochs 1–140 | `1e-3` |
| LR epochs 141–190 | `1e-4` |
| LR epochs 191–200 | `1e-5` |
| 官方 milestone | `140`, `190` |
| AMP skip 上限 | 按完整 200-epoch attempted steps 预先计算的 `0.05%` 总预算 |

这里的 140/190 milestone 语义与原官方 `MultiStepLR` 在每个 epoch 结束后
`step()` 一致：epoch 140 仍使用 `1e-3`，epoch 141 开始使用 `1e-4`；epoch
190 仍使用 `1e-4`，epoch 191 开始使用 `1e-5`。

## 3. 严格确定性与可恢复性

正式进程必须在 CUDA 初始化前设置：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONHASHSEED = '<当前正式 seed>'
```

并启用以下策略：

- `torch.use_deterministic_algorithms(True, warn_only=False)`
- cuDNN deterministic，关闭 benchmark
- 关闭 CUDA matmul/cuDNN TF32
- 关闭 FP16/BF16 reduced-precision reduction
- float32 matmul precision 为 `highest`
- 每个 epoch 使用 `seed + (epoch - 1) * 1009`
- 每个 epoch 重建训练 DataLoader，并把观测 sample order 写入 SHA-256 journal
- 保存 scaler 起点、终点和发生 skip 的零基 batch index，可独立 replay

恢复仅用于同一正式 run 的故障恢复。恢复点必须是原子写入的 `last.pt` epoch
边界；签名、模型、Adam 全状态、GradScaler 全状态、历史和 sample order 都需
重新验证。已经发布 `verification_v1.json` 的 run 不允许恢复或覆盖。

## 4. 预检与 no-clobber

正式预检：

1. 新鲜重哈希已冻结的 16000-row SyncG-train content inventory；
2. 验证 manifest、manifest protocol、pinned VDN commit/model source；
3. 验证官方 ResNet-18 checkpoint SHA-256；
4. 对每个 seed 在 CPU 上重建两次初始模型，要求语义 digest 完全一致；
5. 签署 trainer、protocol、preflight、supervisor、基础 trainer、adapter、
   hardened state protocol、inventory tool、VDN model 和本文档的 source hash；
6. 要求三个正式输出目录全部不存在；
7. 明确记录没有打开 test/public/field/sealed/confirmatory 数据，且预检没有执行
   CUDA tensor 运算。

预检发布后、正式训练前，还必须执行两个相互独立的 Python/CUDA 进程，对固定
seed `20260720` 完整重建模型、Adam、GradScaler 与 DataLoader，并各自完成
epoch 1 的完整训练和 grouped validation。训练/验证 metrics、sample order、
模型/Adam/scaler 语义状态和健康检查必须逐项完全一致，才发布 determinism
authorization。该 probe 是 GPU 工作，同样必须等待人工批准后才能启动。

预检、单 seed verification 和 cohort 文件均使用原子 no-clobber 发布。任何同名
文件已存在时直接失败，不能覆盖。

## 5. 200 epochs 是硬停止边界

200 epochs 是本项目 VDN 官方预算公平复现的终点，不是下一轮自适应延长的检查
点。每个 run 在 epochs 181–190 与 191–200 上报告：

- validation angle MAE 均值绝对相对变化是否 `<1%`；
- validation loss 均值绝对相对变化是否 `<1%`；
- 全局最佳 epoch 是否落在最后 5 epochs。

这些项目仅为 `tail_diagnostic`，不参与 cohort 授权。若任一 seed 尾段仍明显改善
或最佳 epoch 落在最后 5 epochs：

- 仍在 epoch 200 停止；
- 正文如实写入“预算终点仍有改善趋势”的限制；
- 不追加 Phase-4；
- 不对该 seed 单独追加 epochs；
- 不修改 `<1%` 阈值。

cohort 的训练侧完整性依据是：三个 seed 都从头完成完整 200 epochs、预算/调度
相同、预检和 source/content identity 一致、checkpoint/optimizer/scaler/history
验证完整、且不存在数据泄漏。尾段诊断不改变上述终止边界。

## 6. 正式执行顺序

必须使用 PowerShell 7。当前实现完成后应停在 GPU 启动前，待根审查明确批准后
才能执行 `Train`。

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_vdn_official200.ps1 -Mode Preflight

# 只有在人工审查并明确批准 GPU 调度后，先运行两个独立完整 epoch-1 probe：
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_vdn_official200.ps1 -Mode Probe

# Probe 通过并发布授权报告后：
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_vdn_official200.ps1 -Mode Train

& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_vdn_official200.ps1 -Mode Verify

& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_vdn_official200.ps1 -Mode Cohort
```

故障恢复命令仍保持原始 200-epoch 终点：

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_vdn_official200.ps1 -Mode Train -Resume `
  -Seeds 20260720
```

该命令不是追加预算，仅从该 seed 已验证的 epoch 边界恢复到 epoch 200。

## 7. 预期机器制品

- 预检：
  `artifacts/protocols/vdn_official200_preflight_v1.json`
- 两进程完整 epoch-1 确定性授权：
  `artifacts/protocols/vdn_official200_determinism_probe_v1.json`
- 三个 run：
  `artifacts/runs/vdn_syncg_official200/seed_<seed>/`
- 每个 run：
  `last.pt`, `best.pt`, `summary.json`, `verification_v1.json`
- cohort：
  `artifacts/protocols/vdn_official200_three_seed_cohort_v1.json`

只有最终 cohort 报告可以授权冻结的 supporting test evaluation；它始终不授权
field/sealed/confirmatory evaluation，也不授权更多 VDN 训练。
