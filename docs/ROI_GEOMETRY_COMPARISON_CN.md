# YOLO、DeepLab、VDN：效率与 zero-shot 迁移数据

更新日期：2026-09-07。已替换 Industrial 的旧参考点检测器，重跑三种子 DeepLab、VDN 推理，并完成五项 GPU 效率测量。

## Zero-shot 实图迁移

三个读数模型及新的参考点模型均使用有记录的 SyncG 训练与源域验证集，Industrial 不参与本次拟合或参数选择。这里的自动 zero-shot 限定在**给定 ROI**的读数任务，不包含全帧仪表检测。RF100 的 DeepLab、VDN 仍使用人工参考几何，须称为标注辅助的组件迁移。

| 测试集 | 方法 | 六条件 NMAE（%FS） | clean NMAE（%FS） | 六条件 Acc@5（%） | 六条件 Coverage（%） |
|---|---|---:|---:|---:|---:|
| Industrial-1395 | YOLO11s-Pose-4KP | 31.0045 ± 7.6146 | 22.4665 ± 7.0043 | 27.7419 ± 5.8869 | 99.6734 ± 0.2357 |
| Industrial-1395 | DeepLabV3+-ROI + SyncG 参考点 | 32.9356 ± 4.0842 | 19.0515 ± 5.0103 | 40.3943 ± 4.7985 | 85.2011 ± 0.8063 |
| Industrial-1395 | VDN + SyncG 参考点 | 25.3840 ± 6.7070 | 23.9596 ± 5.6809 | 42.5090 ± 5.3783 | 99.6734 ± 0.2357 |
| RF100-VL | YOLO11s-Pose-4KP | 16.1632 ± 4.6280 | 10.9039 ± 5.2046 | 42.5313 ± 12.6235 | 99.7057 ± 0.4179 |
| RF100-VL | DeepLabV3+-ROI | 15.8532 ± 4.0813 | 8.8550 ± 2.0809 | 74.9448 ± 4.4548 | 94.4077 ± 2.5514 |
| RF100-VL | VDN | 13.0129 ± 3.2761 | 11.6135 ± 4.6620 | 62.3620 ± 5.1912 | 100.0000 ± 0.0000 |

Industrial-1395 为三个实拍来源合并的 1,395 张 ROI、52 个采集组，每种子 8,370 行；RF100-VL 为独立的 151 张 ROI、35 个组，每种子 906 行。旧简报中的 1,546 张 Industrial 将 RF100 混入，现已拆开。汇总对样本-条件行等权，不对 Industrial 来源做宏平均。

## 参考点检测器替换

旧权重 `yolo_pointbest.pt` 仅记录相对训练配置 `./datasets/1/meter.yaml`，对应数据清单不可定位，无法排除与 Industrial 重叠。已改用各自同种子的 SyncG YOLO11s-Pose-4KP `best.pt`，只取中心、起点和终点；检测选择和置信度检查也不使用预测指针尖。DeepLab 的方向仍来自分割，VDN 的方向仍来自向量网络。这是两个组合读数流水线，参考点网络的成本计入下方完整效率。

三份 pose 训练清单均已核对：12,866 张训练、1,576 张源域验证，均属于 SyncG，train/val ID 交集为 0。来源证据、使用边界和重跑命令见 [参考检测器审计](ROI_REFERENCE_DETECTOR_AUDIT_CN.md)。

| 读数组件 | 旧参考点六条件 NMAE（%FS） | SyncG 参考点六条件 NMAE（%FS） | Coverage 均值（%） |
|---|---:|---:|---:|
| DeepLabV3+-ROI | 54.2308 ± 1.4463 | 32.9356 ± 4.0842 | 50.1195 → 85.2011 |
| VDN | 54.7987 ± 2.8860 | 25.3840 ± 6.7070 | 52.5926 → 99.6734 |

旧参考检测失败为 3,926/8,370（46.9056%）；新参考检测三个种子的失败分别为 10、30、2/8,370。参考几何成功不等于最终读数成功：DeepLab 仍可能因预测中心与指针分割不匹配等原因失败。新旧差值描述本次替换的实测结果，未另作显著性检验。

## 效率实测

RTX 4060、CUDA 12.8、PyTorch 2.11.0，FP32、batch=1。每臂独立进程，使用同一清单的前 100 张 clean ROI（real_000001–real_000100），预热 20 次；效率仅测 seed 20262020。计入 CPU 预处理、数据传输、全部执行的网络、后处理及 CUDA 同步，排除图像文件解码、模型加载、退化生成和准确率评分。真实输入的额外调用已核实各网络输入均为 float32。

完整的给定 ROI → 读数流水线：

| 模型及计时终点 | 参数量（M） | 神经 GFLOPs* | P50 / P95（ms） | FPS | 峰值显存（MiB） | 有效输出 / 计时次数 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s-Pose-4KP → 读数 | 9.715 | 8.038 | 6.398 / 7.656 | 152.97 | 84.2 | 100 / 100 |
| DeepLab + SyncG 参考点 → 读数 | 50.062 | 99.282 | 19.558 / 20.763 | 51.13 | 271.4 | 97 / 100 |
| VDN + SyncG 参考点 → 读数 | 25.093 | 25.431 | 12.380 / 13.349 | 80.03 | 158.4 | 100 / 100 |

原生组件的额外测量，终点分别为概率图和方向，不包含参考点网络与读数换算：

| 模型及计时终点 | 参数量（M） | 神经 GFLOPs* | P50 / P95（ms） | FPS | 峰值显存（MiB） | 有效输出 / 计时次数 |
|---|---:|---:|---:|---:|---:|---:|
| DeepLab → 指针概率图 | 40.347 | 91.244 | 11.942 / 12.689 | 85.16 | 201.7 | 100 / 100 |
| VDN → 指针方向 | 15.377 | 17.393 | 5.611 / 6.763 | 174.32 | 87.5 | 100 / 100 |

FPS 为 1000 / 平均毫秒延迟，不是 P50 的倒数。所有失败调用保留在计时分母；完整 DeepLab 流水线的这 100 次调用中有 3 次指针组件过短失败。原生组件的“有效输出”不代表读数成功率，也不能用本计时子集代替上方三种子全域 Coverage。

参数量按加载后、Ultralytics 原生融合前的网络统计，组合流水线包括完整 9.715M 参数 pose 参考点模型。输入分辨率为 YOLO/VDN 384、DeepLab 256，与各自训练设置一致。*GFLOPs 使用 PyTorch FlopCounterMode 的受支持 ATen 运算，乘加计为 2 FLOPs；仅为固定尺寸网络前向的受支持运算计数，遗漏预处理、NMS、CPU 几何及不受支持的运算。组合行按两个网络各前向一次求和，不能称为完整端到端 FLOPs，也不与旧 THOP GMAC 数值直接混用。

效率汇总、实际精度和硬件：[公开 JSON](data/roi_comparison_efficiency_public.json)；紧凑数据：[CSV](data/roi_comparison_efficiency.csv)。逐次延迟和完整本地路径保留在本地实验目录。复现每臂：

```powershell
$roiArm = 'deeplab_auto_geometry'
.\.venv\Scripts\python.exe -m experiments.benchmark_roi_comparison_efficiency --arm $roiArm --output-json "artifacts/runs/roi_efficiency_repeat/$roiArm.json" --output-markdown "artifacts/runs/roi_efficiency_repeat/$roiArm.md"
```

其余四臂为 `yolo11s_pose4kp`、`vdn_auto_geometry`、`deeplab_component`、`vdn_component`。自动参考点默认使用同种子 source pose。

## 源域留出结果

SyncG 为同源域中的场景互斥留出，不能作为跨数据集 zero-shot：1,558 张、14 个场景、每种子 9,348 行。此次只重跑 Industrial，以下既有源域结果不变。

| 方法 | 六条件 NMAE（%FS） | clean NMAE（%FS） |
|---|---:|---:|
| GeoPRR-Net | 1.0013 ± 0.0382 | 0.7828 ± 0.0204 |
| YOLO11s-Pose-4KP | 5.5666 ± 2.9705 | 2.1777 ± 2.4698 |
| DeepLabV3+-ROI | 1.7076 ± 0.4078 | 0.1869 ± 0.0695 |
| VDN | 1.6620 ± 0.0878 | 1.1077 ± 0.1715 |

DeepLab 和 VDN 在 SyncG 同样使用标注中心和量程端点进行离线换算。作为 source-only 对照，冻结的 GeoPRR-Net 在 Industrial-1395、RF100-VL 的既有六条件 NMAE 分别为 12.3664 ± 1.8084%FS、5.0098 ± 1.5344%FS。论文仓库另外报告的 Industrial 2.3262%FS 使用目标域有监督 OOF 适配，应与这里的 source-only 协议区分。

## 统计、验证与来源

准确率与误差的 ± 为种子 20262020、20262021、20262022 的样本标准差（ddof=1）；效率为单种子的独立测量。六条件是 clean、blur_moderate、blur_severe、perspective_moderate、perspective_severe、combined_severe。失败保留在分母，normalized absolute error=1.0。NMAE 为归一化绝对误差均值乘 100；Acc@5 阈值为 0.05；Coverage 是成功返回读数的比例。

YOLO 使用固定源域验证集的 best checkpoint（30 epochs），DeepLab 使用源域验证 Dice 的 best checkpoint（20 epochs），VDN 使用 200 epochs terminal checkpoint。训练预算不同，差异不能全部归因于网络结构。VDN 为公开架构在 SyncG 上重训，并非官方预训练模型结果。DeepLab/VDN 准确率重跑沿用评估器的 CUDA autocast，效率统一 FP32。目标域此前已作为回顾性测试查看，本次替换不构成新的盲测集。

新推理包含三份参考几何缓存共 25,110 行，以及 DeepLab/VDN 的六份读数预测共 50,220 行。连同未改动的 YOLO 和 RF100 结果，本次汇总重新计算 83,484 行读数误差，和存储的逐行误差差值为 0；三方法的样本、条件、分组和目标值匹配。几何与计时相关既有测试，以及新增指针尖隔离、坐标映射、置信度测试均通过。

最新三种子目标域汇总：[公开 JSON](data/roi_comparison_zero_shot_public.json) 与 [CSV](data/roi_comparison_zero_shot.csv)，含逐种子、clean/六条件、Acc@2/Acc@5、Coverage 和参考模型来源。公开导出不含本地绝对路径、原图、权重或逐图读数记录。新推理的本地目录为 `artifacts/runs/roi_source_pose_reference_20260907/`，效率目录为 `artifacts/runs/roi_comparison_efficiency_source_pose_20260907/`。旧参考点结果保留于本地 `artifacts/runs/roi_comparison_three_seed_complete/`。
