# 标定与指针状态的组合读数设计

## 研究判断

局部主峰读出是后处理，保留为强解码对照。单独添加 Q/U 两条辅助损失具有可解释的优化作用，但尚不足以认定强创新。主要研究问题是：在同一表盘改变指针状态后，刻度标定 U 能否稳定，并与另一状态图的 Q 正确组合；这种训练能否比等曝光的普通监督和一致性训练带来实际收益。

当前已实现并开始训练首种子 Q/U 辅助损失候选，尚无其完整训练结果。跨状态主机制已有基础配对/warp/组合损失代码，但当前训练没有启用受控组；现有 14,494 条源训练 metadata 没有完整 g/r/a/c 因子记录，不能据场景名称直接认定它们是受控对。

一个同场景四状态 CPU 诊断组已完成渲染及标签组合核验，状态为 0.15/0.35/0.65/0.85。11 次渲染前后及调用前检查的非干预因素均相等，4 个源配对可由现有 loader 读取；16 个 GT 跨图组合的最大读数误差为 `0.000073016%FS`。这是生成链路和标签合同证据，没有使用学习模型，不能代表组合训练或泛化收益。结果见 [state_pilot_20261020](data/geoprr_v2_state_pilot_20261020/README.md)。正式受控源组、多 recipient 采样、留出组合边和等曝光三臂仍待完成。

## 已有工作与可以借鉴的机制

| 工作 | 可借鉴内容 | 当前设计需超出的部分 |
|---|---|---|
| [DARK，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html) | 将坐标编码和分布解码作为完整研究问题 | 固定峰值窗口不单独承担方法贡献 |
| [BCIR，TPAMI 2023](https://arxiv.org/abs/2301.10431) | 分析 softmax/期望读出的偏差、梯度与分布先验 | 不把人体姿态的偏差分析直接当作仪表模型已证原因 |
| [Weakly-Supervised Disentanglement Without Compromises，ICML 2020](https://proceedings.mlr.press/v119/locatello20a.html) | 利用共享部分变化因素的图像对学习表示 | 配对和解耦术语本身不新；需建立读数任务的物理组合关系 |
| [Swapping Autoencoder，NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/50905d7b2216bfeccb5b41016357176b-Abstract.html) | 交换分量以约束结构/纹理分工 | 普通表示交换本身不新；本任务要组合出物理上可检验的读数 |
| [Learning to Read Analog Gauges from Synthetic Data，WACV 2024](https://arxiv.org/abs/2308.14583) | 合成结构监督、分割引导与多头角度读数 | 换轻骨干、多头、几何标签均已有先例 |
| [Under Pressure，ICRA 2024](https://arxiv.org/abs/2404.08785) | 刻度/椭圆、指针、OCR 标定的分工及逐阶段失败分析 | 可解释几何流程已有；其未知量程/OCR任务与归一化读数不同 |

以上检索提供设计依据，尚非完整新颖性审查，不使用“首次”或期刊层级保证。

## 首候选：分别训练 Q 与 U 的读数作用

保留 MobileNetV4、U/Q/M、全部原损失与 65 轮配方。训练时分别将预测 Q 与正确 U/M 组合，将预测 U 与正确 Q/M 组合，目标均为正确 U/Q/M 的完整组合分布。两项前 K−1 项 CDF 距离权重为 1，预测 Q 的 GT 支持缺额权重为 0.1，仅在 5 轮 warm-up 后启用。部署仍只输入 RGB，不新增网络参数。

这一步检验各场是否能独立承担正确读数功能，避免只依赖最终标量允许的误差抵消。相同读数可以对应不同空间位置，因此 Q 的原二维监督仍保留。共享骨干仍可能存在梯度相互影响；标签替换不是因果识别证明。

原种子 20262020 的源验证 NMAE 至少相对下降 5%，且严重误读数和 tail10 均不增加，才进入下一轮作用对照。原期望与固定 r8 均保留；新辅助 loss 下降不等于准确率改善。

## 主要方法候选：同表盘跨状态组合

源组固定真实布局 g、外观 a、相机 c，只改变物理状态 r。每组至少四个状态，保证每个刻度 donor 与多个状态 recipient 重组，并留出未用于组合监督的 donor–recipient 边。参数记录必须证明只有指定因素改变。

```text
同一表盘图 A → U_A / M_A ─┐
                         ├→ 空间组合 → 应读出图 B 的状态
另一指针状态图 B → Q_B ───┘
```

先限定相同相机和 ROI 网格。外观与相机作为留出泛化轴，跨相机组合留作后续扩展；图像 warp 不等同于真实相机重新成像。Q 始终是共同表盘面的虚拟读出点，不能用突出表盘的可见针尖替代。不同布局不直接交换后沿用标签。

当前 loader 每个 sample_a 只允许一个 partner。兑现多 recipient 需要独立实现组采样与留出边；现有原始训练名单保持冻结，新受控组须另存清单和协议，所有对照得到相同新增图像。

## 诊断组入口

官方 SyncG 工程、portable Blender 4.1.1、bundled Python 及独立 OpenCV 存放于项目 `rendering/`。CPU 入口仅为 Blender 子进程加载项目私有的共享库和 Python 包，当前训练 `.venv` 独立使用。

```bash
bash experiments/geoprr_v2_render_cpu.sh --background --factory-startup \
  --python experiments/render_geoprr_v2_state_group.py -- \
  --source-root /root/autodl-tmp/geoprrnet/rendering/vendor/SyncG-337b868f0f5495b31acc383433cac44040247bf2 \
  --output /root/autodl-tmp/geoprrnet/rendering/pilot_state_seed20262020_v1

python -m experiments.prepare_geoprr_v2_state_group \
  --group /root/autodl-tmp/geoprrnet/rendering/pilot_state_seed20262020_v1
```

渲染脚本只加载外部生成器的函数和 imports，使用一次场景构建、随后改变同一指针旋转；生成标签时按实际图像尺寸投影主刻度。全部 RGB/JSON 在同一材质与 world 下完成，分割渲染不进入此路径。原始工程依赖在项目外部资源目录保留，公开代码仅包含适配器及诊断汇总。

此组使用 512×512、16 samples、CPU 4 线程。所有新清单均标记 `diagnostic_only`，不并入原固定清单。正式数据的分辨率、采样质量、执行设备、吞吐与组数需单独确定，不能将该试渲染配置直接当作正式数据协议。

另一个质量与吞吐诊断已排入顺序队列：`python -m experiments.run_geoprr_v2_render_probe`。该入口等待当前 Q/U 读数队列完成及 GPU 无计算进程后，才用独立 Blender 4.5.14/CUDA 生成一个 1920×1080、200 samples、JPEG 95 的四状态组。4.5 路线参考官方 [Blackwell 支持说明](https://developer.blender.org/docs/release_notes/4.4/cycles/)，已验证的 4.1.1 CPU 路径保留。CUDA 场景执行、该设置的图像语义和实际吞吐尚待测量，不能从版本或硬件规格推断速度。

新诊断输出位于 `rendering/gpu_quality_probe_seed20262020_v1/`，继续标记 `diagnostic_only`。它将记录首次和后续渲染时间、整组进程耗时、按秒采样的 GPU 显存与几何合同；这些量分别报告，不把渲染调用时间当完整数据生成耗时。三臂正式训练尚未启动。

## 正式训练前的接口事项

现有 pair 读数损失仅对预测支持质量超过门槛的行计算，GT 有效但预测低支持的组合仍会退出该项。支持缺额惩罚不等于保留该行读数监督。当前无配对的 Q/U 候选不走此路径；正式组合训练需先明确定义覆盖全部 GT 有效边的稳定训练目标，并在留出评估中将低支持边计为失败、保留完整分母。该修改属于新组合协议，不静默改动已有实验。

多 recipient 可通过边展开的数据条目实现：每个条目仍返回一个 donor 和一个 recipient，无需新增网络或多维前向。四状态组可用双向四边环的 8 条训练边，留出两条对角线的 4 条有向边；每个 donor 有两个训练 recipient，留出反向边也不参与训练。各组状态应连续抽样并独立固定边划分，避免只学习全数据共用的四个读数类别。该留出只检验已见图像的未监督组合，不称为未见状态或独立场景泛化。

完整场景快照按组保存一份即可；训练清单应引用组级真实因素记录，避免在每张图和每条边中重复网格与节点数组。三臂须同时记录唯一图像数、边条目数、附加 recipient 暴露及优化步数，避免把重复 donor 计为新增独立图像。

## 决定贡献的对照

在相同 RGB、标签、有效 batch 图像数、前向次数与优化步数下比较：

1. 两图分别做原结构与读数监督；
2. 在前者上添加 U/M/Q 的适用一致性关系；
3. 在前两者上添加跨状态组合读数。

只有第三项优于第二项，才支持组合算子的额外作用；三项相近而共同提高时，收益应归于成组图像暴露或普通一致性。还应检验留出组合边、U 对状态的稳定性、Q 对状态变化的响应及是否绕过结构路径。

## 证据边界

最终需同时报告三种子平均/样本标准差、严重误读修复与新增失败、尾部/最大误差、采集组区间，以及完整训练和推理成本。只有合成组合指标改善时，结论限于表示诊断；真实冻结迁移未改善时，不宣称解决真实域泛化。历史已查看的真实来源明确作为开发/探索性证据，新的确认性主张需要独立保留来源或批次。
