# GeoPRR V2：冻结best64的外部评估

2026-10-22。冻结的v7 full best64在Industrial-1395和RF100-151上的NMAE分别为 **21.226570%FS** 和 **12.691067%FS**。这组结果不支持该模型具有充分的跨数据集泛化能力。两个人口均完整计分，未出现输入失败或无效预测，未训练或适配目标数据。

## 结果

误差均为占满量程百分比；Acc为完整人口中的百分比。

| 人口 | 张数／组数 | NMAE %FS | Median %FS | P90 %FS | Worst10% mean %FS | Max %FS | Acc@1／2／5 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Industrial全体 | 1395／52 | 21.226570 | 14.496543 | 50.752475 | 63.219116 | 99.337763 | 7.7419／12.6165／25.4480 |
| RF100全体 | 151／35 | 12.691067 | 2.948505 | 38.430762 | 53.320607 | 86.109657 | 22.5166／43.7086／57.6159 |

| Industrial固定子集 | 张数 | NMAE %FS | Acc@1／2／5 % |
|---|---:|---:|---:|
| Test-A | 434 | 21.123774 | 13.8249／20.7373／32.9493 |
| Test-B | 814 | 21.789570 | 5.5283／10.0737／23.8329 |
| External-ROI | 147 | 18.412497 | 2.0408／2.7211／12.2449 |

Industrial误差>1／2／5%FS分别为1287／1219／1040张，最大误差ID为`field_xm2_r00329_86f268c9c15d`；RF100分别为117／85／64张，最大误差ID为`rf100_test_000023`。完整逐图、组及子集统计随包保留。

独立CPU核对PASS：1546张的ID与顺序、物理量归一化到FP32 target、逐图误差及全人口／子集／87组的5760个统计字段均一致，最大数值残差为0。两套子进程均正常退出。该核对未重跑模型。

与旧GeoPRR的人口匹配比较、输入流程核对、固定8张源样本运行时重放和典型失败分析见[外部诊断报告](../geoprr_v2_external_diagnosis_20261022/README.md)。旧历史执行像素同一性尚未复核，其限制单独保留。

## 冻结模型与成本

执行使用事先选定的v7 full best64、epoch64、seed20262020、7,461,316参数、497个state tensors。源开发1506 NMAE为0.059146231063%FS。原`RoleTriangleReader`严格加载权重，输入仅已有ROI RGB，沿用768／192、ImageNet归一化及checkpoint AMP设置，输出原生全质量读数。目标按原FP32约定计分，单图失败error=1，不从分母剔除。

| 成本 | Industrial | RF100 |
|---|---:|---:|
| 实际ROI CNN图像前向 | 1395 | 151 |
| 模型／CNN调用 | 88／88 | 10／10 |
| 失败占位前向 | 0 | 0 |
| 反传／optimizer更新 | 0／0 | 0／0 |
| 输入准备、传输、前向及收集 wall / s | 16.012784 | 9.662870 |
| 同步前向 wall / s | 11.571161 | 9.097634 |
| Peak allocated / bytes | 1,267,317,248 | 1,265,482,240 |

batch16、workers4，RTX PRO 6000 Blackwell；不排除首次调用开销。这是各数据集一次执行的批处理成本，不是稳态单图延迟基准。模型构造及权重加载不计入表中的输入到输出wall；完整子进程起止时间在各summary的execution中。

## Limitations

Industrial使用既有仪表ROI及标量／归一化读数标签；RF100目标由四点标注离线派生，并非官方标量读数benchmark。标注几何未输入模型。这些结果描述冻结模型在既有ROI人口上的表现，不表示全图自动检测到读数的端到端效果。两套数据的历史结果曾被查看，本次不是新盲测；仅一个冻结checkpoint、一个seed，不给出统计显著性。RPM-10K的1797人口尚缺真实全图检测器权重及来源，或完整ROI与pass／fail sidecar，未运行，也未套用历史1775／22计数。外部结果不用于选择模型或调参，不追加训练。

## 文件

- `industrial_summary.json`／`rf100_summary.json`：显式allowlist的模型配置、全人口／组／子集结果及成本。
- `industrial_predictions.csv.gz`／`rf100_predictions.csv.gz`：完整归一化逐图结果。
- `industrial_groups.csv`／`rf100_groups.csv`：组级统计。
- `execution.json`：执行与公开字段边界；`cpu_interface_check.json`：RGB预处理、失败分母及真实checkpoint严格加载检查。
- `independent_final_audit.json`：独立CPU逐图与执行核对。

RGB、权重、ROI原路径、原始几何及private trace不在公开包中。
