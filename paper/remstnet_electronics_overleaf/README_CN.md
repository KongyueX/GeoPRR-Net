# ReMSTNet Electronics 最新稿件

这是仓库 `paper/` 下当前公开的 LaTeX/Overleaf 稿件包，按
`MDPI_template_ACS` 模板整理。目前包含：

- 英文题目；
- 英文 Abstract；
- 英文 Introduction；
- Materials and Methods（任务、数据、SARN-v2、ReMSTNet-v3、损失、统计协议）；
- Results（六条件主表、VDN 全量标注参考同像素对比、1,395 ROI 工业实景基线、RPM-10K 原始骨干诊断、153 张工业全图标量诊断、独立 OCR 到物理读数实验、2×2 消融、18 条件压力、自然重复、CUDA 效率）；
- Discussion 与 Conclusions；
- 正式论文图及新增 OCR 端到端部署图；正文引用的 8 张图均提供 PDF
  和 PNG；
- 当前正文所需参考文献；
- MDPI `Electronics` LaTeX 类及全部 `Definitions/` 依赖；
- 已编译的作者工作稿 `remstnet_electronics_manuscript.pdf`。

旧稿、内部审稿记录、Python 缓存、未引用的渲染格式、现场原始图片和
中间数据仍保留在本地工作区，不进入公开仓库。

## 上传与编译

1. 将本目录打包为 ZIP 后上传至 Overleaf，并选择“创建新项目”；
2. 将主文件设置为 `manuscript.tex`；
3. 编译器使用 `pdfLaTeX`；
4. Overleaf 会自动运行 BibTeX 生成参考文献。

## 投稿前仍需填写

- 作者、单位、邮箱和通讯作者；
- Author Contributions、Funding 和 Conflicts of Interest；
- 现场数据的 Data Availability 与图片使用权限；
- Methods 与 Acknowledgments 中 Codex 的实际产品版本、使用日期及完整使用范围；
- 代码、清单和机器可读结果的最终公共存档链接；
- 在 Overleaf 完成最终 `pdfLaTeX + BibTeX` 版式检查，并由全体作者核对正文数字与数据授权。

全文数字依据 ReMSTNet-v3 三种子最终结果，并已区分 ImageNet 初始化、14,442 张 SyncG 基础模型拟合数据与其中 6,616 张 ReMST 模块拟合子集。VDN 对比使用两套既定留出名单的 129 张、6 场景交集：标注中心与有序量程端点仅用于把 VDN 方向换算成进度，因此全条件 774/774、投影条件 387/387 均覆盖；ReMSTNet 相对 VDN 的 NMAE 降幅分别为 47.81% 与 47.99%，但这是组件级而非自动系统级比较。工业实景基线以 `C:/pointer_read/unified_real_photo_progress_v1` 为准，包含 1,395 个标量标注 ROI、52 个组，其解码像素集合与三个既有 FieldGauge 源清单的并集完全一致；六条件总体、投影池和三个投影条件的配对区间均支持 ReMSTNet。RPM-10K 包含模糊与倾斜图像，但当前协议关闭了关系输入，因此只作为原始骨干跨域诊断。另有 153 张去重现场原图的缓存检测器标量诊断，覆盖 151/153；在同一批照片上又独立执行了从工业全图、ReMSTNet 进度预测、现有 PP-OCRv4 与自动量程恢复到物理读数的端到端路径，OCR 权重未经现场拟合。默认解码器输出 27/153，在有标签照片中输出 9/33，条件 NMAE 为 0.036814±0.004015，同一 9 张使用真实量程时为 0.034336±0.003848。全分母 NMAE 为 0.737313±0.001095，说明当前主要瓶颈是 OCR/自动量程覆盖率。该实验沿用现有 OCR，不主张 OCR 模型创新，也不是实时检测器或独立盲测队列。
