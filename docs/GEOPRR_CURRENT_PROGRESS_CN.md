# GeoPRR 当前研究进度

更新时间：2026-10-26T01:58:01.810635+00:00。执行已因旧服务器预算不足暂停，自动heartbeat为PAUSED；换设备后由真人明确恢复。当前没有新的模型可替代既有参照，没有中断训练需要恢复。

## 最新完整结果

最新实验是source-only条件view梯度控制，两个15E阶段均结束，整体`NOT_RETAINED`。critic在线准确率从82.6055%降至46.5975%，预定critic效应通过，但读数mean/tail/max/>5和准确图保护两套联合门槛全部失败。不能以critic接近chance称泛化成功。

| Source-ID292/73组 | Mean %FS | Tail30 %FS | Max %FS | >5% 图数 |
|---|---:|---:|---:|---:|
| identity15参照 | 2.5431218244 | 11.8009553314 | 53.7970364094 | 16 |
| 新detached control | 2.6759538619 | 11.6840900977 | 46.8278169632 | 25 |
| 梯度反转candidate | 3.3223066845 | 16.0971364006 | 81.1839252710 | 38 |

Identity原准确102图中11图新增>5%，control原准确92图中6图新增>5%。Candidate CF按规则未执行，null/0calls；不能用control或identity的CF代替。此控制没有对Industrial/RF做新评估。

唯一fresh独立保存数字审计PASS：42230checks/0fail，15780float对比maxabs1.4210854715202004e-14。正式165756CNN-point图/2595calls，164880critic examples/2580calls；2580backattempt=2579update+1正常AMPskip。Whole supervisor996.849506秒，独立pref/metadata和各计时边界详归档，不能把wall当GPU kernel时间。

[最新完整归档](data/geoprr_v2_conditional_view_adversary_source_control_20261026/README.md)；[审计报告](data/geoprr_v2_conditional_view_adversary_source_control_20261026/completed/audit_report.md)。科研代码03981213603beea957989547690af73004617844，启动/见证5e8a0da6c7ef0970068d5abcd704b29bcd636604。

## 已保存的外域参照

下表来自已核对的V31比较原表，单位均为%FS，tail是各数据集最差ceil(10%)图的均值。没有补评旧模型。不同版本训练数据、输入分辨率、结构/监督和训练历史不同，不是统一预算的因果对比。

| 保存模型 | Industrial1395 mean / tail10 / max | RF151 mean / tail10 / max |
|---|---:|---:|
| 旧GeoPRR历史clean | 11.643618 / 43.950870 / 75.324667 | 6.607079 / 32.748645 / 60.504565 |
| V16 | 11.686054 / 45.734073 / 86.481965 | 10.164776 / 44.210096 / 54.413070 |
| V24 | 23.980845 / 65.664809 / 95.865870 | 7.726820 / 25.325275 / 50.892561 |
| V26 | 22.861047 / 64.456899 / 86.018354 | 9.059529 / 51.670712 / 70.405841 |
| V28 | 12.758458 / 44.068174 / 71.893692 | 10.831387 / 54.135926 / 68.525833 |
| V29 | 22.354576 / 78.790409 / 97.788346 | 11.805282 / 51.893965 / 72.905040 |
| V30 | 15.418735 / 56.544002 / 80.651760 | 12.095891 / 50.749786 / 68.491405 |
| V31 | 37.249295 / 50.422275 / 50.422275 | 23.395320 / 49.616268 / 50.422275 |

没有单一checkpoint在所有指标上统一最优，也没有批准promotion。V16保留为近期轻量source-selected参照；旧GeoPRR historical clean保留为历史参照，其pixel/fullweight链未重新认证。不能拼不同checkpoint的单域最佳，更不能把论文主模型的多条件/三seed数字和本表的single-clean人口混为同一排行榜。

[完整保存比较原表](data/geoprr_research_handoff_20261026/saved_external_comparison.csv)包含更多版本和source范围；[V31完整报告](data/geoprr_v2_groupnorm_semantic_20261025/README.md)。V27未完成epoch/外评，不能编造baseline。

## 已关闭的近期路线

- V31 GroupNorm近常数源学习失败；source60/best1及两外评均封存，不恢复。
- C3原小控制有部分收益，但matched identity15联合比较不支持；不改order/seed/budget救过。
- Mean-tail15和固定Sobel-structure30均未通过各自联合读数保护。
- State-change q监督在第14个quartet的必要data gate前停止；52partial maps不能作为完整标签消费，不删失败组或改阈值重试。
- 当前conditional-view对抗控制的critic目标通过、读数失败；不改alpha/width/conditioning/seed/预算救过。
- 旧physical-visible监督、固定parent appearance replay、font/prior/roll诊断、G1/teacher/anchor/VDN/virtualQ/配对守恒等已有负面和资格全部保留，不换名字重跑。

## 迁移与继续入口

- [设备迁移交接](GEOPRR_DEVICE_MIGRATION_HANDOFF_CN.md)
- [可复制的新设备续接提示词](GEOPRR_CONTINUE_ON_NEW_DEVICE_PROMPT_CN.md)
- [更高效的分阶段训练与早停规则](GEOPRR_EFFICIENT_ITERATION_RULES_CN.md)
- [机器可读状态](data/geoprr_research_handoff_20261026/progress.json)

## Limitations

这份进度是现有结果的归档与续接，不是新训练或新GT认证。单seed/反复查看的source与外域开发人口不支持显著性、新盲测或新理论；C3原149/148/1 raw-label派生资格不清PASS。最新迁移SSH连接不可用，当前远端进程/文件状态无法重新核验；最后已证实研究父子全退出且GPU空闲的时间是2026-10-26 00:57:52UTC。未执行平台停机/释放、私有资产传输或目的设备验证。
