# 同类方法可复现性审计（2026-07-21）

本文必须区分“内部消融”和“外部同类方法”。Geometry-v1/v2、融合、残差及门控
都属于本文内部变体，不能代替外部模型对比。下面记录提交论文前实际核验过的公开资源，
避免把论文中声称“将发布”的代码或权重误写成已复现结果。

| 方法 | 核验提交 | 公开代码 | 官方权重 | 当前处理 |
|---|---|---|---|---|
| VDN / Pointer-10K | `DrawZeroPoint/VectorDetectionNetwork@68afe1e` | 有，包含训练、推理和数据格式 | 无；README 的模型下载链接为空 | 唯一务实的外部视觉基线；必须重新训练并明确写作 `VDN (our retraining)` |
| Learning to Read Analog Gauges from Synthetic Data (WACV 2024) | `fuankarion/automatic-gauge-reading@a7d5956` | 无；仓库只有 25 字节 README | 无 | 只放相关工作，不伪造复现成绩 |
| DialBench / MRLM | `Event-AHU/DialBench@f97093c` | 有训练/benchmark 框架，但多处配置仍是作者私有绝对路径 | README 明确为 `TBD / coming soon` | 作为单独 VLM 协议；不能和本文已知量程的 RPM 子集主表直接混比 |

公开入口：

- [VDN 官方仓库](https://github.com/DrawZeroPoint/VectorDetectionNetwork)
- [WACV 2024 方法声明的官方仓库](https://github.com/fuankarion/automatic-gauge-reading)
- [DialBench 官方仓库](https://github.com/Event-AHU/DialBench)

## 最小可发表比较方案

正文主对比保留以下四类即可：

1. 传统几何读数（无学习）；
2. 项目原始 Transformer；
3. VDN 的公开架构重训版；
4. 本文完整方法。

其余内部方法移到消融表。VDN输出的是指针向量，不直接输出标量读数；公平接入时应让它与
本文共享同一个表盘检测、起终刻度和已知量程适配器，并同时报告其原生方向角误差和适配后的
NMAE/Acc@2%。必须注明共享组件，不能写成未经修改的官方端到端成绩。

## 数据和训练约束

- Pointer-10K 官方入口目前是百度网盘，许可为 CC BY-NC-SA 4.0；Hugging Face Hub
  检索未发现可核验的同名镜像。若使用第三方镜像，必须先用官方文件清单或哈希核验内容。
- 若VDN用SyncG train重训，主表应写 `VDN architecture, retrained on SyncG`，并与本文使用
  完全相同的SyncG train/test以及控制退化图像。
- 若VDN用Pointer-10K重训，则写 `VDN, retrained on Pointer-10K`，结果属于跨域迁移设置；
  不可和同域SyncG训练方法含混表述。
- 在外部基线真正跑完以前，只能声称相对项目原始模型和内部消融改善，不能声称SOTA。

