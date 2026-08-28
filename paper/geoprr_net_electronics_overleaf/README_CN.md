# GeoPRR-Net / Electronics Overleaf 上传包

直接将本目录整体上传至 Overleaf，并把 `manuscript.tex` 设为 Main document。

目录内容：

- `manuscript.tex`：论文主文件；
- `references.bib`：参考文献；
- `Definitions/`：仓库内的 MDPI *Electronics* 模板文件；
- `figures/`：正文图件、生成图件所用的汇总 CSV 与脚本；图 1 正文直接引用 PNG，中文审阅稿也只引用 PNG。
- `figures/assets/`：图 1 所用公开 RF100-VL 示例 ROI 及其来源说明。

投稿前请在 `manuscript.tex` 中搜索 `to be confirmed`，由作者补齐：

- 单位、城市、国家和通讯邮箱；
- 作者贡献；
- 基金信息；
- 利益冲突声明；
- Industrial-1395 的访问条件和最终预测账本 URL；
- RF100-VL 派生目标清单与预测账本的最终公开存放地址；
- 作者姓名、顺序、通讯作者以及 AI 使用声明。

GeoPRR-Net 的公开代码与复现实验脚本位于 `https://github.com/KongyueX/GeoPRR-Net`。Industrial-1395 在整篇论文中只作为一个完整的 1,395 图像、六条件、test-only 队列，不包含任何拆分结果。新增的 RF100-VL 评测使用 151 张公开测试图像、35 个保守源组和预先由检测标注导出的归一化目标；它被明确限定为外部迁移检查，不作为官方标量读数榜单。`research/`、整篇论文的预编译 PDF 和其他编译中间文件均未放入上传包。
