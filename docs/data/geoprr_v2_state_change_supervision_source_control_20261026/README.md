# Source state-change supervision control

**执行已完整结束；科学结论为 `NOT_ADMITTED_SOURCE_OBSERVATION`。** 第 14 个 source quartet（0-based index `13`）未通过冻结的必要 gate `all_six_pairs_above_quantization_unit`，整个训练 population 保持 `admitted=false`，完整可接受 target cache 为 **0**。本轮在 CPU source 准备阶段正常科学拒绝，未进入真实模型构造、preflight 或任一训练 arm。

研究状态仍为 **ACTIVE**。本轮 GPU 执行为 0，当前没有下一项模型运行；不消费保留的 partial cache，不删失败组或仅训练通过的 13 组 prefix，也不调整 threshold、λ、target、预算、seed 或重试以求通过。本轮没有模型成绩或机制收益结论。

科学修订 `85821b55653fbbb6e374c4ec700cae61055d7066` 已冻结并推送。保存的 [launch.json](launch.json) 是启动时 metadata 的显式投影：`2026-10-25T23:18:47.849380+00:00` 启动 parent `438380`。Preparation `438381` 与 finalize `438405` 两个 CPU stage 均 complete/exit 0；最后一个 stage 于 `2026-10-25T23:18:54.003722+00:00` 完成，最终 queue 更新于 `2026-10-25T23:18:54.074029+00:00`。保存的 `2026-10-25T23:19:54.498473+00:00` completion/idle witness 显示实际 `438380/438381/438405` 均已退出、GPU compute rows 为空。

## Source 准备结果

| 保存证据 | 实际结果 |
|---|---:|
| 计划 source train rows / quartets | 5496 / 1374 |
| Train manifest reads | 1 |
| 已处理 quartets | 14 |
| 局部必要 gate 通过 quartets | 13 |
| 首个失败 quartet | 0-based 13，第 14 组 |
| Source RGB decode calls | 56 |
| 独立 native-header 访问 | 56 |
| Quartet operator calls | 14 |
| 已写 partial target maps | 52 |
| 整体 source population admitted | false |
| 接受的完整 target cache | 0 |
| 预分配 partial target 文件 stat 大小 | 5628032 bytes |

56 次 RGB decode 与 56 次独立 header 访问是不同访问边界，不能把 decode 数写成全部文件打开数。目标文件在循环前预分配，`5628032 bytes` 不代表 5496 个合法 target。部分结果保留且不消费；私有 partial 与 preparation-completed 记录均维持 `admitted=false`，下游 loader 的准入 guard 在打开 target array 之前拒绝该结果。

## 未执行的模型阶段

| 阶段或指标 | 保存状态 |
|---|---|
| 真实模型 / weights / preflight | 未构造、未读取、未执行 |
| Correct arm / cyclic arm | 均未运行 |
| 每个 arm 已训练 epochs | 0 |
| CNN / point / auxiliary calls | 0 / 0 / 0 |
| Combined backward / optimizer updates / GPU calls | 0 / 0 / 0 |
| Source03、ID/OOD、真实 Industrial/RF 与 external RGB/model execution | 0 |
| Preflight / correct / cyclic / trained-model result / comparison | null |
| Native / CF metrics | null / null |
| 真实架构 metadata | 234 tensors / 465 states / 77 BN 仅为预期，未观察 |

冻结流程原计划是 CPU 准备全体通过后才运行一次 preflight，再进入 correct 15E；correct 全部 native/条件 CF 联合门槛通过后才运行 cyclic 15E。此次整体 source 准入失败直接进入 finalize；计划预算不计作实际训练。已有 necessary synthetic CPU 证据不能写成真实 runtime preflight PASS。

## 实际成本

| 时间/内存边界 | 保存值 |
|---|---:|
| Preparation module entry → result metadata | 3.860509477555752 s |
| Preparation supervisor child interval | 4.499085124582052 s |
| Decode + native preprocessing | 0.7790854945778847 s |
| Target construction，含失败 operator | 1.7908196970820427 s |
| Header + alignment | 0.01877351850271225 s |
| Preparation peak RSS，native Linux KiB | 709124 |
| Finalize module entry → result metadata | 1.0332649238407612 s |
| Finalize supervisor child interval | 1.592880416661501 s |
| Whole supervisor process，最终 queue 保存时 | 6.21332449093461 s |
| Whole controller final numeric notice | 6.2135750241577625 s |
| Launch timestamp → final queue timestamp | 6.224649 s |
| Point-only time | null |
| GPU time | 0.0 s |

`completed.runtime.full_entry_seconds=1.0332649238407612s` 属于独立 finalize child，不是整个 pipeline 用时。Module-entry、supervisor child、whole-process、final notice 与 timestamp interval 的边界分别保留，未知 point 用时保持 `null`。

## 已完成审计与公开文件

唯一 fresh、独立非作者的 standard-library saved-number/execution CPU audit 结论为 **integrity `PASS`**：**30/30 checks PASS、26/26 numeric comparisons PASS**。整数精确比较，浮点绝对容差为 `1e-12`。科学结论仍为 `NOT_ADMITTED_SOURCE_OBSERVATION`，`scientific_model_result=null`。

本目录只发布 6 个显式审计结果/证据文件；CSV 保持 plain CSV，仅规范为 LF，数字字符串原样保留。本次公开投影不重新计算指标、target 或 gate。

| 内容 | 公开文件 |
|---|---|
| 最终结论与范围 | [audit_result.json](completed/audit_result.json)、[report.md](completed/report.md) |
| 逐项保存检查 | [checks.json](completed/checks.json) |
| 保存数值比较 | [numeric_comparisons.csv](completed/numeric_comparisons.csv) |
| 安全 metadata 投影 | [collected_metadata.json](completed/collected_metadata.json) |
| Completed 显式文件清单 | [archive_allowlist.json](completed/archive_allowlist.json) |

根目录完整公开清单见 [public_allowlist.json](public_allowlist.json)。冻结的 [protocol.json](protocol.json)、[config.json](config.json)、[cpu_checks.json](cpu_checks.json) 与 [中文协议](../../GEOPRR_V2_STATE_CHANGE_SUPERVISION_SOURCE_CONTROL_CN.md) 保持原样。原始 run、完整 contracts/populations/row-cycle bindings、target arrays、RGB、weights 及私有路径保留在私有执行上下文。

## Limitations

- 失败 pair 的原始 RMSE 未在拒绝前保存。本轮只能核验保存的 reason/witness，无法独立数值验证该失败 pair；不补算或虚构 RMSE。
- 13 组仅满足保存的局部必要 gate witness，52 张 partial target map 不能代表整个 1374 组/5496 图 population 准入；预分配文件不是完整可消费 cache。
- `prepared.json` 按完整 metadata 文档 parse，其 nested train/ID/OOD contract 私有复制；只有 train manifest 行用于构建 q。不能把整个文件读取称为 strict train-only；复制的既有 scalar labels 是 metadata，没有新计算 physical-y/GT。
- Image-derived source spatial observation 是原 scalar-y 之外的额外 source 监督。残差可能包含阴影、JPEG、反射、遮挡与其他 state 残影，不是 physical needle、scale、visible mask 或 GT geometry 认证。
- 真实架构与 preflight witness 未构造；234 trainable tensors、465 states、77 BN 是冻结预期。Necessary synthetic CPU 证据不证明真实 runtime metadata 或真实 preflight PASS。
- 此前 C3 仍为 149 checks / 148 PASS / 1 unavailable derivation-field FAIL；历史 pixel/full-weight 与 physical-label derivation 未重新认证，不改写为完整 PASS，也不推断标签冲突。
- 单 seed、重复使用的 source 开发数据与计划人工 CF 不支持新 blind test、显著性、因果机制或真实域泛化；source03/真实域 RGB、模型与选择均未执行。
- 公开包保存审计结果与比较证据；原执行检查依赖私有 run 上下文，不能从本公开包独立重跑。Integrity PASS 仅覆盖保存数值/执行边界，不产生训练成绩或机制收益。
