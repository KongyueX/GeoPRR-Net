# 可复制到新设备对话的续接提示词

填写设备和路径后，把下面整段发送给新的工作对话。不要自动恢复旧机heartbeat。

```text
继续GeoPRR-Net的source-only结构/表示研究。仓库是https://github.com/KongyueX/GeoPRR-Net，使用现有main，不创建分支。

新设备：[本机或新SSH连接]
工作根：[WORK_ROOT，下面有code/datasets/rendering/diagnostics/runs/cache]
仓库路径：[REPO_ROOT]
项目Python：[PYTHON]
私有资产位置：[原source RGB、manifest/ROI/labels、ImageNetcache、checkpoints的位置]
资源限制：[可用GPU、单次时长/预算；不购买算力]

先读取Git中的docs/GEOPRR_CURRENT_PROGRESS_CN.md、docs/GEOPRR_DEVICE_MIGRATION_HANDOFF_CN.md、docs/GEOPRR_EFFICIENT_ITERATION_RULES_CN.md、docs/data/geoprr_research_handoff_20261026/progress.json，再按需读最新completed审计。私有refine-logs若已迁移可作补充，不能假设它在Git里。不要全量回读所有历史，也不要重跑completed任务。

旧服务器因费用不足已停止继续工作，自动化geoprr-v23-zero-shot为PAUSED。只有我指定新设备并明确恢复后才允许执行；不连接旧地址启动训练，不擅自重启heartbeat、租服务器或买算力。先确认新设备和私有资产就绪，保留封存manifest，明确路径映射；缺数据/cache就列缺项，不能重渲染、换weights或假称已经迁移。

当前没有新模型被提升为默认，也没有中断训练要resume。最新conditional-view控制已完整结束：critic训练准确率82.6%→46.6%，但GRL source-ID mean/tail/max为3.3223/16.0971/81.1839%FS，>5为38；identity15为2.5431/11.8010/53.7970，>5为16，原准确102图有11图变>5。结果NOT_RETAINED，GRL CF按gate未运行，真实Industrial/RF没有新评估。V16与旧GeoPRR历史参照继续保留，不拼不同checkpoint的单域最好。所有旧负结果、C3原149/148/1资格及历史pixel/fullweights未重认证边界保留。

研究目标是整体mean、严重tail/max、新增大错与原准确样本共同改善，不能只优化某一项或critic代理指标。只根据source证据提出一个真正不同的可推翻问题；旧q、Sobel、失败GRL、G1/teacher/anchor/VDN/virtualQ/配对守恒等不改名重试，不调阈值/seed/数据子集救过。

提高效率：不要默认最低60轮。新增观测/标签方法先完成最小CPU/metadata人口可用性准入，再写完整训练pipeline。新模型先最小必要CPU和source0opt见证；通常1–3轮检查是否真的学主任务，明显loss不降、预测近常数且source验证无改善时就停，不能用进程健康/梯度finite作为继续理由。有明确源学习证据才按事前冻结的5–10轮小cap继续，正式长训需另有依据和总预算。预算是上限不是必须花完；不是所有模型3轮应收敛，早期波动和best不更新也不能单独判死刑。训练必须在运行前写清source-only earlystop、最小改善、patience、tail/max/新增>5和准确集保护，正常AMPskip如实保留不补更新。

设计/代码/必要检查和保存数字审计尽量在CPU环境完成，避免付费GPU空转等文档或代理。优先复用已有可靠数据/runner，只改新增方法必要部分，不复制大量训练代码或重复旧tests/GT/pref。每次只保留一个实现owner和必要的唯一fresh非作者完整审计，不加多轮review。

Industrial1395和RF151绝不用于训练、无标签adapt、总体统计fit、BNadapt、checkpoint/epoch/earlystop/LR/训练超参/ROI/threshold/温度选择，也不据其病例设计labels。正式候选先source完整结束锁定source-best，再各外域一次固定评估，最后ONEfresh保存数字CPU核对与explicitallowlist归档。Source/单域好不叫整体成功；单seed已看开发域不作显著性/新盲测/新理论。

先给我简短的新设备就绪状态和唯一下一问题、早停/最大预算/验收规则；已授权且合同完整后自行推进，不为常规可逆操作反复问许可。没有可靠依据就明确保持GPU0，不捏造模型或phantom结果。不要删除旧资产、改其他项目或系统Python/CUDA/conda、碰officialtest4000、上传私有RGB/weights/features/contracts/训练身份数组/privatepaths、发送第三方消息。
```
