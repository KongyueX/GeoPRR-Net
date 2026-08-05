# PointerMeterReaderFastAPI

## Research release: PEPD

The research method is **pivot estimation and probabilistic direction (PEPD)**.
It predicts a pivot, fuses a direct direction vector with a circular posterior,
estimates angular uncertainty, and uses projectively paired ray supervision during
training. The failure-aware dual-representation router (FADR) is retained as a
pre-specified transfer audit, not as part of the final PEPD deployment method.

### Main aggregate results

All reading failures remain in the denominator with normalized error 1. The
training comparison uses the same 4,380-row/197-group SyncG-training grouped OOF
union. The field row is a prediction-independent sensitivity analysis retaining
725 images/18 groups after two complete groups linked to development-set physical
meters were excluded.

| Method | SyncG-train OOF NMAE | Field sensitivity NMAE |
| --- | ---: | ---: |
| PEPD | 0.148497 | **0.025465** |
| VDN official-200 reproduction | **0.148212** | 0.126295 |
| Original Transformer | 0.209035 | 0.230224 |
| Base-mask geometry | 0.115905 | 0.117155 |
| PEPD + FADR | **0.088168** | 0.065463 |

PEPD and VDN are statistically tied on the partial grouped SyncG-training OOF
union. On the retained field sensitivity set, VDN minus PEPD NMAE is 0.100829
(95% group-bootstrap CI 0.059141--0.147008), and PEPD reaches 96.83% Acc@5%.
Because the physical-entity exclusion and PEPD-only deployment decision were made
after the original one-shot execution, this is descriptive sensitivity evidence;
a new prospectively entity-disjoint field cohort is still required for strict
confirmation.

The three-seed mechanism audit supports fused decoding and projective pairing,
especially under severe perspective degradation. It does not show an independent
angle-accuracy gain from the explicit equivariance term. FADR's three-seed,
five-variant feature-family ablation stayed in a narrow 0.088088--0.089510
source-domain NMAE range, yet the frozen router degraded in the field. This
negative transfer is the reason FADR remains an audit rather than the final model.

### Research code map

- PEPD model and objectives:
  [`experiments/probabilistic_pivot_direction.py`](experiments/probabilistic_pivot_direction.py),
  [`experiments/pepd_uncertainty_objectives.py`](experiments/pepd_uncertainty_objectives.py), and
  [`experiments/projective_circular_transport.py`](experiments/projective_circular_transport.py).
- PEPD training and mechanism audit:
  [`experiments/train_pepd_convergence_syncg.py`](experiments/train_pepd_convergence_syncg.py),
  [`experiments/train_pepd_mechanism_continuation_syncg.py`](experiments/train_pepd_mechanism_continuation_syncg.py), and
  [`experiments/summarize_pepd_mechanism_evaluations.py`](experiments/summarize_pepd_mechanism_evaluations.py).
- VDN reproduction:
  [`experiments/train_vdn_official200.py`](experiments/train_vdn_official200.py),
  [`experiments/evaluate_vdn_official200_train_oof.py`](experiments/evaluate_vdn_official200_train_oof.py), and
  [`experiments/vdn_baseline.py`](experiments/vdn_baseline.py).
- FADR audit:
  [`experiments/train_joint_nested_fadr.py`](experiments/train_joint_nested_fadr.py),
  [`experiments/fadr_feature_sets.py`](experiments/fadr_feature_sets.py), and
  [`experiments/verify_fadr_multiseed_cohort.py`](experiments/verify_fadr_multiseed_cohort.py).
- Aggregate evaluation and resource scripts:
  [`experiments/summarize_field_physical_entity_leakage_sensitivity.py`](experiments/summarize_field_physical_entity_leakage_sensitivity.py),
  [`experiments/benchmark_final_model_stack_resources_v3.py`](experiments/benchmark_final_model_stack_resources_v3.py), and
  [`experiments/build_paper_figures.py`](experiments/build_paper_figures.py).

Create the pinned Python 3.11 environment and run the data-free checks with:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/requirements-training.lock.txt
.venv/Scripts/python.exe -m unittest `
  test.test_probabilistic_pivot_direction `
  test.test_pepd_convergence_protocol `
  test.test_projective_circular_transport `
  test.test_vdn_official200
```

Datasets, field records, per-sample predictions, fitted weights, private
configuration, and the manuscript are not distributed in this repository. Formal
commands fail closed unless their local protocols and artifact hashes match. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance and
redistribution boundaries. The repository does not yet grant a project-level
software license; source visibility is not permission to reuse or redistribute.

## 推荐基础方案（测试请使用这个）

当前推荐的基础方案是：**`ransacFun` 图像校正 + 几何读数 + 残差修正 + 门控回退**，对应配置为 `correction_mode=ransacFun`，`reading_backend` 从下面两种残差修正方案中任选一个。

- **几何读数**：先用 `geometry_direct`（指针角+起终点角+量程）和 `geometry_direct_v2`（鲁棒针尖投票）各算一次，`geometry_fusion` 取两者平均，作为几何主链结果，两种残差修正方案都建立在这个结果之上。
- **门控回退**：残差修正失败、门控拒绝或几何主链本身失败时，都会自动按 `geometry_fusion` → `geometry_direct` → `geometry_direct_v2` 的顺序回退，全程无需人工干预；如需在几何整体失败时进一步回退到 transformer，额外传 `"geometry_fallback_to_transformer": true`。
- 排查问题时打开 `return_reading_details: true`，返回中的 `reading_details` 会带上各候选后端的详细结果，方便确认是否触发了回退（`fallback_from` 字段）以及残差修正是否生效（`calibration` / `delta_pred` 字段）。

### 方案 A：树模型残差校准器 `geometry_fusion_calibrated`（默认推荐）

用标量特征（几何角度、mask 点数、椭圆代理特征等，不直接读图像像素）训练的树模型预测残差，并带 learned-gate 门控（残差过大或门控判定不可靠时自动拒绝，退回 `geometry_fusion`）。

```json
{
  "modelName": "PointerMeterInferModel",
  "inferData": "data/img_000001.png",
  "dataType": "filepath",
  "inferConfig": {
    "scaleStart": 0,
    "scaleEnd": 1.6,
    "correction_mode": "ransacFun",
    "reading_backend": "geometry_fusion_calibrated",
    "return_reading_details": true
  }
}
```

旧的 38 样本 learned-gate 校准器属于本地工程产物，公开仓库不分发；只有该旧路径实际存在时才会自动加载。论文方法 `geometry_fusion_weighted_calibrated` 则会查找正式流水线生成的 `artifacts/runs/syncg_full/calibrator.joblib`。也可以显式指定校准器文件：

```json
{
  "correction_mode": "ransacFun",
  "reading_backend": "geometry_fusion_calibrated",
  "residual_calibrator_path": "path/to/calibrator.joblib",
  "return_reading_details": true
}
```

传 `"residual_calibrator_path": ""`（空字符串）可显式禁用默认校准器，行为退化为普通几何融合。

验证数据（未公开的旧工程 learned-gate 树模型校准器，仅作回归记录）：

| 数据集 | 后端 | MAE | RMSE | 最大误差 |
| --- | --- | ---: | ---: | ---: |
| `ground_truth.xlsx` 30 张复核 | `geometry_fusion_calibrated@ransacFun` | 0.008126 | 0.015969 | 0.058200 |
| `ground_truth - val.xlsx` 8 张验证 | `geometry_fusion_calibrated@ransacFun` | 0.002017 | 0.003492 | 0.009141 |
| 同一 8 张验证集基线 | `geometry_fusion@ransacFun` | 0.014490 | - | - |
| 同一 8 张验证集基线 | `transformer@ransacFun` | 0.018415 | - | - |

### 方案 B：CNN 残差模型 `geometry_hybrid` / `geometry_hybrid_gate`（备选，谨慎使用）

直接输入裁剪表盘灰度图 + 指针 mask + 骨架图，用 CNN（`ResidualHybridNet`）预测残差和置信度。`geometry_hybrid_gate` 在此之上再加一层指针角度窗口门控（角度落在 `[45.14°, 52.09°]` 之外时不采用 CNN 结果，直接回退到 `geometry_fusion_calibrated` / `geometry_fusion`）。

```json
{
  "modelName": "PointerMeterInferModel",
  "inferData": "data/img_000001.png",
  "dataType": "filepath",
  "inferConfig": {
    "scaleStart": 0,
    "scaleEnd": 1.6,
    "correction_mode": "ransacFun",
    "reading_backend": "geometry_hybrid_gate",
    "return_reading_details": true
  }
}
```

旧版会在本地工程产物目录查找 `best_model.pt`，该权重不在公开仓库；复现方案 B 时必须显式提供 `residual_hybrid_model_path`。`residual_hybrid_max_abs_delta`（默认 `0.05`）控制门控阈值：CNN 预测的残差绝对值超过该阈值时视为不可信，自动回退到几何融合结果，避免残差模型在陌生数据上把稳定的几何结果改坏。

```json
{
  "correction_mode": "ransacFun",
  "reading_backend": "geometry_hybrid_gate",
  "residual_hybrid_model_path": "path/to/best_model.pt",
  "residual_hybrid_max_abs_delta": 0.05,
  "return_reading_details": true
}
```

验证数据（38 条真实读数全量）：

| 后端 | MAE | RMSE | 最大误差 |
| --- | ---: | ---: | ---: |
| `geometry_hybrid_gate@ransacFun` | 0.018919 | 0.025678 | 0.058200 |
| `geometry_fusion_calibrated@ransacFun` | 0.019207 | 0.025678 | 0.058200 |
| `geometry_hybrid@ransacFun`（无角度门控） | 0.031069 | 0.037701 | 0.066143 |

注意：`geometry_hybrid` 不带角度门控，误差明显更大更不稳定，不建议直接使用；`geometry_hybrid_gate` 加了角度门控后分数和方案 A 接近，在整体 38 条数据上略好一点。

### 如何选择

- 默认建议用**方案 A（`geometry_fusion_calibrated`）**。它的训练特征是抽象的几何/mask 统计量，对分割模型预测 mask（而不是人工标注 mask）的适应性更好，已经在两批独立数据集（30 张复核 + 8 张 val）上验证过。
- **方案 B（CNN，`geometry_hybrid_gate`）** 直接吃图像和 mask 像素，训练数据主要来自人工标注 mask，线上实际用的是分割模型预测的 mask，存在训练/推理不一致的风险（团队内部验证记录里也是这么说的）。想对比效果时可以显式切换到这个后端跑一遍，但**外部测试如果只能选一个方案，请用方案 A**。
- 两个方案可以在同一批图片上分别跑，配合 `return_reading_details: true` 对比 `resultNum` 和 `calibration` / `delta_pred` 字段，判断哪个更适合你的实际拍摄条件。

## 其他可选 reading_backend

`reading_backend` 还支持 `geometry_direct_v2`、`geometry_fusion`、`geometry_fusion_weighted`：

- `geometry_direct_v2`：实验性鲁棒几何针尖投票，一般不单独使用，仅作为融合分量。
- `geometry_fusion`：`geometry_direct` 和 `geometry_direct_v2` 的平均值，是上面两个残差修正方案共同的基础。
- `geometry_fusion_weighted`：根据两种针尖估计的轴线一致性和 mask 支持度做质量加权；保留 `geometry_fusion` 作为简单均值消融。
- `geometry_fusion_weighted_calibrated`：在质量加权结果上应用归一化残差和选择性门控。训练与公开数据评测见 [`experiments/README.md`](experiments/README.md)。

论文最终方法现已通过显式后端 `reference_conditioned_final` 接入生产 API。它不会改变
原有默认后端，且要求显式 artifact manifest、冻结 SyncG 分割权重以及严格的制品/源码
哈希审计；配置、返回字段和失败规则见
[`docs/REFERENCE_CONDITIONED_PRODUCTION_CN.md`](docs/REFERENCE_CONDITIONED_PRODUCTION_CN.md)。

如需获取与论文最终方法完全相同的概率方向专家原始输出，可显式选择
`reading_backend="probabilistic_vector"`（别名 `raw_probabilistic_vector`）。该后端复用
同一次表盘框、置信度、起始角、量程角和参考分支，只调用冻结方向模型，不运行 base
calibrator、reference-conditioned calibrator 或 router 参与读数计算（完整 bundle
仍会加载并审计）。响应直接给出 `prediction`、
`progress`、`direction`、`pointer_angle`、`uncertainty` 和 `artifact_audit`；为保持原始
数值，`auto_zero` 必须为 `false`。它与最终后端共用显式 manifest 和源码/权重哈希审计。

这是一个基于 FastAPI 的指针式表计读数识别服务。接口通过 `PointerMeterInferModel` 完成表盘检测、可选图像校正、指针分割、读数推理和结果图返回。

## 启动与接口

服务配置位于 `config/app_config.yaml`，可通过其中的 `host` 和 `port` 修改监听地址。

可在服务启动前用环境变量选择论文实验生成的分割权重和推理设备；不设置时使用仓库自带权重与 CPU：

```powershell
$env:POINTER_METER_SEGMENTATION_WEIGHTS = "artifacts\runs\syncg_segmentation\best.pt"
$env:POINTER_METER_DEVICE = "cuda"
python main.py
```

主要推理接口为 `/infer`，请求中的 `modelName` 固定使用：

```json
"PointerMeterInferModel"
```

`inferConfig` 用于控制表计量程、图像校正、读数偏移、默认角度、mask 校验和返回图片。

## PointerMeterInferModel inferConfig

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `infer_mode` | string | `infer` | 推理模式，目前使用 `infer`。 |
| `scaleStart` | number | `0.0` | 表计最小读数。 |
| `scaleEnd` | number | `1.6` | 表计最大读数。 |
| `confidence` / `conf` | number | 不传 | YOLO 检测置信度阈值，不传时使用模型默认值。 |
| `use_origin_when_no_meter` | bool | `false` | 未检测到表盘时是否使用原图继续推理。 |
| `auto_zero` / `auto_zero_reading` | bool | `true` | 是否启用自动归零。开启后，当最终读数小于阈值时返回 `0.0`。 |
| `auto_zero_threshold` / `zero_threshold` | number | `0.025` | 自动归零阈值。当 `result < auto_zero_threshold` 时置为 `0.0`。 |
| `reading_backend` / `reading_method` / `meter_reading_backend` | string | `transformer` | 读数后端。除原有 transformer/geometry 系列外，论文生产后端包括 `reference_conditioned_final` 和只返回未经校准、未经路由方向输出的 `probabilistic_vector`（别名 `raw_probabilistic_vector`）。两者均需显式冻结 manifest；完整配置见生产接入文档。 |
| `residual_calibrator_path` / `geometry_calibrator_path` / `calibration_model_path` | string | 不传 | 树模型 residual 校准器 joblib 路径。旧 `geometry_fusion_calibrated` 只查找旧工程产物；论文后端 `geometry_fusion_weighted_calibrated` 只查找正式流水线生成的 `artifacts/runs/syncg_full/calibrator.joblib`，避免把基于不同融合基线训练的残差模型混用。传空字符串 `""` 可显式禁用默认校准器。 |
| `residual_hybrid_model_path` / `geometry_hybrid_model_path` / `hybrid_model_path` | string | 不传 | 方案 B（`reading_backend="geometry_hybrid"` 或 `geometry_hybrid_gate`）使用的 CNN 残差模型权重路径（`.pt`）。公开仓库不包含旧 `best_model.pt`，复现时需显式提供。 |
| `residual_hybrid_max_abs_delta` / `geometry_hybrid_max_abs_delta` / `hybrid_max_abs_delta` | number | `0.05` | 方案 B 的门控阈值。CNN 预测残差绝对值超过该值时不采用，自动回退到几何融合结果。 |
| `return_reading_details` / `return_backend_details` | bool | `false` | 是否在返回中追加 `reading_details`，包含 transformer、geometry_direct、geometry_legacy 等各候选后端的详细对比信息，可用来确认残差修正是否生效、是否触发回退。 |
| `geometry_fallback_to_transformer` / `fallback_geometry_to_transformer` | bool | `false` | 当几何相关后端全部失败时，是否自动回退到 transformer 结果。 |
| `result_pointer_image` | bool | `false` | 是否返回结果图 base64。结果图包含中心点、起点、终点和参考线。 |
| `result_mask_image` | bool | `false` | 是否返回指针 mask 图 base64。业务接口返回的 mask 不叠加点位或调试箭头。 |

### 图像校正参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `correction_mode` | string | `ransacFun` | 图像校正算法。可选 `off`、`ellipse`、`ransacFun`、`ransacFunbackup`、`square`、`stretch`。兼容别名 `image_correction`、`correction_algorithm`；`ellipse` 兼容别名 `auto`、`dial`、`circle`、`deskew`。 |
| `stretch_x_ratio` / `horizontal_stretch_ratio` | number | `1.0` | `correction_mode="stretch"` 时生效，直接对裁剪表盘原图进行水平方向缩放。例：`1.2` 表示宽度放大 20%。 |
| `stretch_y_ratio` / `vertical_stretch_ratio` | number | `1.0` | `correction_mode="stretch"` 时生效，直接对裁剪表盘原图进行垂直方向缩放。例：`0.9` 表示高度压缩到 90%。 |

`correction_mode` 说明：

- `off`: 关闭图像校正，直接使用检测裁剪出的表盘图。
- `ellipse`: **表盘面椭圆自动矫正 + YOLO中心点约束（推荐，无需人工提供拍摄角度）**。在裁剪表盘上用亮度+低饱和锁定白色表盘面并拟合椭圆的**长短轴和角度**，同时用YOLO检测表盘中心点（classId=0）作为椭圆**中心约束**，构建混合椭圆后用仿射把椭圆"压回"正圆以消除俯仰/偏航带来的前缩。含质量校验（轴比/占比/居中不达标则回退原图）与正表盘保护（椭圆轴比 ≥ 0.95 视为已够圆，跳过矫正）。实测YOLO中心命中率80%，对斜拍表盘可显著降低读数漂移并消除极端翻车样本。
- `ransacFun`: 使用基于起点、终点、中心点的仿射校正。
- `ransacFunbackup`: 使用备用透视校正算法。
- `square`: 将裁剪表盘直接 resize 拉伸为正方形，边长为 `max(width, height)`，不补边。
- `stretch`: 按 `stretch_x_ratio` 和 `stretch_y_ratio` 直接缩放裁剪表盘，不强制正方形。

### 读数与默认角度参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `reading_offset` / `endNum_offset` | number | `0.0` | 在 transformer 得到的 `endNum` 基础上增加偏移值，再按 101 个刻度取模后参与读数计算。用于微调读数偏差。 |
| `start_end_distance_threshold` / `same_point_distance_threshold` / `point_same_distance_threshold` | number | 自动 | 起点和终点距离小于等于该阈值时视为同一个点。默认值为表盘宽度的 3.5%，且不小于 10 像素。 |
| `start_end_position` / `start_end_layout` | string | `start_left_end_right` | 起点和终点的左右位置关系。可选 `start_left_end_right`、`start_right_end_left`。 |
| `default_start_angle` | number | `45.0` | 起点和终点都未检测到时使用的默认起点角度。 |
| `default_range_angle` | number | `270.0` | 默认量程角度。只检测到起点或终点时，用该角度推算另一个端点；起终点都缺失时也用该角度。 |
| `snap_out_of_range_pointer` / `clamp_out_of_range_pointer` / `snap_pointer_when_out_of_range` | bool | `true` | 指针计算结果超过量程时，是否吸附到更近的起点或终点值。 |

### Mask 主轴校验参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `validate_mask_line` / `check_mask_line` | bool | `true` | 是否校验指针 mask 的主轴延长线经过表盘中心附近。 |
| `mask_center_threshold_ratio` | number | `0.10` | mask 主轴到表盘中心的最大允许距离比例，阈值为 `min(width, height) * mask_center_threshold_ratio`。 |

当 `validate_mask_line=true` 时，程序会对最大连通域 mask 使用 `cv2.fitLine` 拟合指针主轴。如果主轴延长线距离表盘中心过远，认为指针 mask 寻找错误，接口返回：

```json
{
  "status": false,
  "message": "无法找到指针",
  "result": null,
  "result_pointer_image": null,
  "result_mask_image": null
}
```

## inferConfig 示例

### 使用 square 强制正方形校正

```json
{
  "scaleStart": 0,
  "scaleEnd": 1.6,
  "correction_mode": "square",
  "auto_zero": true,
  "auto_zero_threshold": 0.025,
  "result_pointer_image": true,
  "result_mask_image": true
}
```

### 使用 stretch 拉伸校正并增加读数偏移

```json
{
  "scaleStart": 0,
  "scaleEnd": 1.6,
  "correction_mode": "stretch",
  "stretch_x_ratio": 1.15,
  "stretch_y_ratio": 0.95,
  "reading_offset": -1,
  "auto_zero": true,
  "auto_zero_threshold": 0.025,
  "validate_mask_line": true,
  "mask_center_threshold_ratio": 0.12
}
```

### 配置默认起终点角度

```json
{
  "default_start_angle": 45,
  "default_range_angle": 270,
  "start_end_position": "start_left_end_right"
}
```

## Quick Browse 调参工具

`utils/angleDetect/quick_browse_results.py` 提供 Tkinter GUI，用于快速对同一张图比较不同校正算法，并编辑可直接用于 `PointerMeterInferModel` 的 inferConfig。

运行示例：

```powershell
python utils\angleDetect\quick_browse_results.py D:\datasets\pointer_meter\images
```

工具特性：

- 每张图按行对比 `off`、`ransacFun`、`ransacFunbackup`、`square`、`stretch`。
- 每行显示原图、结果图、mask 图和推理状态。
- 右侧可实时编辑 `stretch_x_ratio`、`stretch_y_ratio`、`reading_offset`、默认角度、mask 校验阈值等参数。
- 默认保存配置到 `<image_root>/quick_browse_inferconfig.json`。
- 配置按排序后的图片相对序号保存，可用于另一个同顺序图片目录中复用同一组参数。

指定配置路径：

```powershell
python utils\angleDetect\quick_browse_results.py <image_root> --config <config_json_path>
```

## 返回结果

`/infer` 成功时返回：

```json
{
  "status": true,
  "message": "归一化指针位置为..., 表盘读数是...",
  "result": 1.23,
  "result_pointer_image": "base64或null",
  "result_mask_image": "base64或null"
}
```

失败时 `status=false`，`message` 会说明失败原因，例如 `未检测到表盘` 或 `无法找到指针`。

## 手工测试 `/infer`

启动服务后打开 `http://127.0.0.1:30600/docs`，在 `POST /infer` 中点击 `Try it out`。

### 使用相对路径

```json
{
  "modelName": "PointerMeterInferModel",
  "inferData": "data/KZT_HD_20260616150323688.jpg",
  "dataType": "filepath",
  "cameraTimeout": 10,
  "inferConfig": {
    "scaleStart": 0,
    "scaleEnd": 1.6,
    "result_pointer_image": true,
    "result_mask_image": true
  }
}
```

### 使用 Windows 绝对路径

```json
{
  "modelName": "PointerMeterInferModel",
  "inferData": "D:\\datasets\\pointer_meter\\sample.jpg",
  "dataType": "filepath",
  "cameraTimeout": 10,
  "inferConfig": {
    "scaleStart": 0,
    "scaleEnd": 1.6
  }
}
```

返回中的 `result` 为最终表盘读数；`result_pointer_image` 和 `result_mask_image` 为可选的 base64 结果图。

## 批量跑 `data/` 目录

可直接运行：

```bash
python batch_infer_260520_to_excel.py
```

默认会：
- 读取仓库下 `data/` 目录中的图片
- 调用本地 `http://127.0.0.1:30600/infer`
- 生成 `data_infer_results.xlsx`
- 保存返回的结果图到 `data_infer_result_images/`

也可以显式指定路径：

```bash
python batch_infer_260520_to_excel.py --image-dir data --output data_infer_results.xlsx --result-image-dir data_infer_result_images
```
