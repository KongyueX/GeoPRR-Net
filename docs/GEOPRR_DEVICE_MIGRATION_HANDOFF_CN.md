# 换设备继续研究：交接说明

2026-10-26：因旧服务器费用不足，真人要求保存进度并换设备。自动heartbeat `geoprr-v23-zero-shot` 已设为 **PAUSED**；不要连接旧地址自动启动或恢复任何训练。新设备由真人指定并明确恢复后，再依次进行环境/私有资产核验和新方法准入。

## Git已保存什么

[当前进度](GEOPRR_CURRENT_PROGRESS_CN.md)、[机器可读状态](data/geoprr_research_handoff_20261026/progress.json)、历史冻结协议/config、公开聚合结果与审计证据、实验源码均在现有main。最新conditional-view两arm已完成，结果NOT_RETAINED；source critic变得接近chance，读数却未过mean/tail/max/大错/准确集保护。没有新模型替换基线，没有中断训练需要resume。

[新设备可复制提示词](GEOPRR_CONTINUE_ON_NEW_DEVICE_PROMPT_CN.md)和[效率/早停新规则](GEOPRR_EFFICIENT_ITERATION_RULES_CN.md)是之后工作的入口。不要直接执行任何已completed的旧runner。

## 远端核验边界

最后已有实际witness为2026-10-26 00:57:52UTC：研究父子均退出、GPU compute为空。本次迁移metadata SSH连接未成功（exit255），所以当前远端进程、文件存在/大小、实时环境均 **未重新核验**，不宣称现在已关机或数据已备份。没有执行平台停机/释放、删除文件或私有资产传输。

[静态私有资产清单状态](data/geoprr_research_handoff_20261026/private_asset_inventory_status.json)列的是代码/协议中已有相对位置，presence/size为null，不是存在性证明。旧实例是否运行和是否仍计费，应在实际平台核实；Git提交与暂停自动任务不能代替实例停机。释放旧实例前务必先确认私有数据已保全。

## 私有资产必须单独迁移

下面路径均相对私有工作根，不包含旧SSH地址或绝对个人路径。不要将它们上传到公开Git。

| 资产 | 位置/用途 |
|---|---|
| Source及其他私有datasets | `datasets/`，保留manifest、scalar标签、ROI和原数据结构；officialtest4000保持隔离不评测 |
| 已有原source RGB | `rendering/compositional_source_v1/`、`rendering/generalization_endpoints_v11_20261023/` |
| 已有roll source | `rendering/generalization_roll_v22_20261024/`，保留train/validation manifests与declaredmetadata |
| 冻结的小source packet | `diagnostics/geoprr_source_needle_holdout_control_population_20261025/`，含5496train/292ID/220sourceOOD的私有绑定；不能将OOD拿来训练 |
| 训练结果与完整状态 | `runs/`，保留best/latest/last、optimizer/scaler/RNG/history、logs、privatecontracts和所有正负结果 |
| 既有ImageNet cache | 项目cache下的HF/Torch缓存，特别是MNV4 `mobilenetv4_conv_medium.e500_r224_in1k`；不能换成新foundationweights |
| Renderer/源资产和原快照 | 现有`rendering/vendor/`及必要tools/assets；迁移是保存，不是授权重渲染/旧parent replay |
| 本地私有研究记录 | 原工作区`refine-logs/`按需私有复制，不能git add；Git中的公开handoff足以定位，详细contract/CSV/资格在私有补充中 |

优先完整保存已有私有资产。若只能迁出子集，至少先保护原source RGB/manifest/labels/ROI、既有ImageNetcache、V16和历史参照checkpoints及当前全部实验metadata；任何删减或释放旧资产都需要真人明确指定，不能用“最新模型不保留”作为删除失败权重的理由。本次没有下载这些资产，目的设备容量和复制完整性尚未核验。

## 新设备目录和环境

建议保持工作根下`code/`、`datasets/`、`rendering/`、`diagnostics/`、`runs/`、`cache/`的相对结构，Git仓库放在`code/`。实际位置由真人提供；不要照抄旧服务器地址/端口或绝对路径。

原manifest/private contract可能嵌入旧绝对路径。保留封存原文件，在新工作副本中建立明确的新旧root映射，并核对sample IDs、row order、scalar标签、ROI和所需文件引用。不能直接全局字符串替换原封存contract后宣称bitwise相同，也不能把新随机render当旧数据恢复。没有未完成run要续训；之后是独立新version/protocol，而不是恢复completed checkpoint任务。

最后保存的runtime见证：Torch `2.11.0+cu128`、CUDA runtime `12.8`、GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`。这是旧机历史记录，不是当前或新设备已验证环境。依赖定义见[requirements](../experiments/requirements-geoprr-v2.txt)和[training lock](../experiments/requirements-training.lock.txt)。不要直接复制旧`.venv`里的CUDA二进制到不兼容设备；在项目隔离环境中核对新GPU/driver/Python兼容性，不改系统Python/CUDA/conda或其他项目，不安装/下载新foundationweights来替换缺失cache。

新设备只做会影响下一行动的必要检查：私有文件引用/数量和source绑定、环境与device可用性、下一候选所需的最小新CPU及source0opt输入/梯度/dtype见证。不要以迁移为由重跑全部旧tests/GT/模型外评。必要的硬件兼容smoke不是新benchmark，必须单列0optimizer成本；没有新方案时不要启动真实模型。

## 下一阶段的科学边界

没有当前已批准的新模型方向。原RGB source参照可学，多个已关闭控制没有整体收益；先用既有源证据定义一个真正不同的可推翻问题，不能降低旧q阈值/只训13组、调整已失败GRL的alpha/width/conditioning/seed/预算、重开Sobel或旧GT-role/teacher/anchor/配对守恒来救过。

两个真实外域Industrial1395/RF151继续完全排除训练、无标签adapt、总体统计fit、BNadapt、epoch/checkpoint/earlystop/LR/超参/ROI/threshold/温度选择。模型候选先通过source mean/tail/max/新增大错/准确集联合保护，正式source完整结束并锁定source-best后，各外域固定评一次，再ONEfresh非作者保存数字CPU核对。公开Git只收explicitallowlist结果，rawRGB/weights/features/contracts/训练身份数组/privatepaths留私有。
