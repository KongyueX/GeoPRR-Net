# Industrial 参考点检测器来源核查

核查日期：2026-09-07。来源核查范围为当前工作区的相关训练脚本、数据配置、已有审计、训练清单及本地 checkpoint 元数据，使用 CPU 读取。替换后的 GPU 读数及效率实测另见 [实验简报](ROI_GEOMETRY_COMPARISON_CN.md)，本次没有重新训练模型。

## 旧检测器：训练数据仍不可确定

对 `utils/angleDetect/yoloDetection/result/yolo_pointbest.pt` 使用 CPU 读取 checkpoint，核实以下内嵌字段：

| 字段 | 值 |
|---|---|
| `date` | `2025-09-05T11:06:33.355978` |
| `version` | `8.3.193` |
| `train_args.data` | `./datasets/1/meter.yaml` |
| `train_args.model` | `yolo11s.yaml` |
| `train_args.task` | `detect` |
| `train_args.epochs` | `200` |
| `train_args.batch` | `64` |
| `train_args.imgsz` | `640` |
| `train_args.seed` | `0` |
| `train_args.name` | `train` |
| `train_args.project` / `save_dir` | 均未提供可定位原训练目录的值 |
| 模型类别 | `0: one`、`1: two`、`2: there` |

当前工作区没有 `datasets/1`，在工作区相关配置、YOLO 源码目录及本地数据目录的限定检索中，未找到对应的 `meter.yaml`。`git log --all -- utils/angleDetect/yoloDetection/result/yolo_pointbest.pt` 也没有返回该文件的历史记录。相对路径不能确定原训练机器或数据位置，训练日期和类别名不能证明是否包含 Industrial 图像。

以上与本地第三方来源审计中记录的数据来源未知一致。结论是**不能确认旧检测器训练数据，也不能排除 Industrial 重叠**；这不是已经发现重叠。旧 DeepLab/VDN Industrial 结果不能据此升级为整条 ROI 读数流水线的严格 zero-shot 结果。

## 可追溯的替代：现有 SyncG 四关键点模型

替代权重为 `artifacts/runs/roi_comparison_pilot/seed_<seed>/yolo11s_pose4kp/ultralytics/weights/best.pt`。本次读取了每个种子的 `training_summary.json`、`dataset/train.txt` 和 `dataset/val.txt`，并将清单中每个样本 ID 与 `artifacts/manifests/syncg_train.jsonl` 对照：

| 随机种子 | 训练样本 | 源域验证样本 | train/val ID 交集 | 不在 SyncG train manifest 的 ID | epochs | best.pt |
|---|---:|---:|---:|---:|---:|---|
| 20262020 | 12,866 | 1,576 | 0 | 0 | 30 | 存在 |
| 20262021 | 12,866 | 1,576 | 0 | 0 | 30 | 存在 |
| 20262022 | 12,866 | 1,576 | 0 | 0 | 30 | 存在 |

每种子的 14,442 个训练/验证 ID 均对应 SyncG 源域清单；记录的初始化为 `yolo11s-pose.pt`，输入尺寸为 384，checkpoint 选择为 `Ultralytics best fixed-inner-validation fitness`。[训练入口](../experiments/train_yolo11s_pose4kp.py)加载 SyncG train manifest，使用 `_matched_geoprr_partition` 划分训练与内部验证集，再[生成四关键点数据](../experiments/yolo11s_pose4kp.py)。这证明的是本项目有记录的训练与验证来源；不应写成“模型从未使用过任何预训练数据”。

## 替代方案的输入与使用边界

[参考几何适配器](../experiments/roi_reference_geometry.py)按训练定义，将四点解释为 `[pivot, pointer_tip, reference_start, reference_end]`，只取索引 **`[0, 2, 3]`**：

- 检测框选择和置信度检查也只使用这三个参考点；预测指针尖的坐标和置信度不参与选择、成功判定或输出。
- 对已给定 ROI 的条件图像进行 384 × 384 双线性缩放，再将参考点按 `width / 384`、`height / 384` 分别映回原 ROI。
- DeepLab 的指针方向仍来自分割，VDN 的指针方向仍来自方向网络；两者只接收上述自动中心和量程端点。
- 使用替代检测器后必须重新执行参考点检测和受影响的读数评估。旧预测与旧效率数字不能仅通过更名变成新结果。

新方案可描述为“给定 ROI，使用 SyncG 上训练的参考几何与读数组件进行目标域迁移”。它不包含上游全帧仪表检测，也不改变 RF100 现有标注辅助结果的口径。参考点失败和读数失败仍须保留在总分母中。

CPU 行为测试位于 [test_roi_reference_geometry.py](../test/test_roi_reference_geometry.py)，覆盖指针尖完全隔离、非方形 ROI 坐标映射，以及低参考点置信度失败。

## 重现本次替换

三个种子分别使用同种子的 pose 检测器、DeepLab 和 VDN。下面展示单种子命令；将 `$roiSeed` 分别设为 `20262020`、`20262021`、`20262022`，并使用新的输出目录重跑。阈值沿用既有 YOLO ROI 评估设置，没有根据 Industrial 标签调整。旧参考点权重及原始评测保留。

```powershell
$roiSeed = 20262020
$roiRun = "artifacts/runs/roi_source_pose_reference_20260907/seed_$roiSeed"
.\.venv\Scripts\python.exe -m experiments.evaluate_missing_roi_direction_baselines prepare-geometry --output "$roiRun/geometry.jsonl" --reference-detector-weights "artifacts/runs/roi_comparison_pilot/seed_$roiSeed/yolo11s_pose4kp/ultralytics/weights/best.pt" --reference-detector-kind source_pose4kp --image-size 384 --confidence 0.05 --keypoint-confidence 0.05 --max-det 5 --batch-size 32 --device 0
.\.venv\Scripts\python.exe -m experiments.evaluate_missing_roi_direction_baselines evaluate --method deeplab --checkpoint "artifacts/runs/roi_comparison_pilot/seed_$roiSeed/deeplabv3plus_roi/best.pt" --seed $roiSeed --datasets field_gauge_roi_test_a field_gauge_roi_test_b field_gauge_external_roi --geometry-cache "$roiRun/geometry.jsonl" --output-dir "$roiRun/deeplabv3plus_roi_auto_geometry/field" --batch-size 16 --device cuda:0
.\.venv\Scripts\python.exe -m experiments.evaluate_missing_roi_direction_baselines evaluate --method vdn --checkpoint "artifacts/runs/geoprr_vdn_matched/seed_$roiSeed/last.pt" --vdn-source artifacts/vendor/VectorDetectionNetwork --seed $roiSeed --datasets field_gauge_roi_test_a field_gauge_roi_test_b field_gauge_external_roi --geometry-cache "$roiRun/geometry.jsonl" --output-dir "$roiRun/vdn/field" --batch-size 16 --device cuda:0
```

三种子均完成后，重算准确率汇总；RF100 从原标注辅助评测读取：

```powershell
.\.venv\Scripts\python.exe -m experiments.summarize_roi_comparison_zero_shot --missing-root artifacts/runs/roi_source_pose_reference_20260907
```
