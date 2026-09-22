# GeoPRR 补充实验追踪

| ID | 工作 | 状态 | 完成判据 |
|---|---|---|---|
| R0 | 实际full入口、construction、增强、内层fold、availability、模块宽度／预算、smooth-max温度及系数记录 | 待恢复运行记录 | 配置与明确入口可查，缺项逐一列出；已有定义说明不替代实际运行配置 |
| E14 | 独立warm及受影响下游训练 | 待训练 | 三seed独立初始化、完整预测及依赖更新 |
| E27 | 简单双视图头与基础专家消融 | 待训练 | 三个控制头、base/full诊断、完整条件结果 |
| E20 | 两种简单gate | 待训练 | 六个gate运行与uniform/fixed对照 |
| C4 | 增强匹配Raw B0 | 条件触发 | R0确认需补时执行三个seed |
| C13 | OOF内层隔离 | 待核对，必要时重做 | 实际组分配和选模日志；最多60个共同head |
| A-native | Industrial native七指标与20,000次group CI | 暂缓 | 离线文件已保留，不纳入本轮论文修改 |
| A-Fig3 | SyncG图3的scene-bootstrap区间 | 现有结果已完成 | `data/revision_20260922/figure3_*`源表和图形导出；后续checkpoint变化影响对应预测时刷新 |
| A-expert | 候选读数、uniform、entropy逐条件汇总 | 部分诊断已有，待补齐逐行候选与权重缓存 | 不重训专家，完整分母和三seed汇总；已有alpha扫描与集中度保留 |
| A-moment | 末端投影残差与可选耗时 | 部分诊断已有，待统一 | 同checkpoint相同裁剪目标的数值残差；耗时实测 |

配套实现已提供：`revision_full_training` 与 unified 的 official_full 模式、`revision_cache`、`revision_controls`、`revision_diagnostics`、条件触发的 `revision_raw_baseline` 和 `revision_oof_heads`。代码通过 CPU 构造数据测试不代表正式训练完成。跨环境执行见 `REVISION_EXECUTION_PROMPT_20260922.md`。

本清单没有启动新增训练。训练状态必须由后续实际运行更新。Industrial native专项保持暂停，不列为本轮E14训练链的验收条件。
