# 配对守恒参考更新：冻结正式训练与匹配对照合同

日期：2026-10-22。用户已明确授权正式候选65轮；通过原11项门槛及best48／best64保护后，才运行匹配global对照65轮。唯一模型seed20262020，不增加loss、不扩seed，所有现有模型、partial和失败结果保留。

## 方法、初始化与匹配关系

候选为配对守恒参考更新，匹配对照为同信息global更新。沿用best64的MobileNetV4／FPN、中心／SPD、A—P／A—B／P—B与全质量progress读数。共享有序9方向token核144→32→1、无输出bias，新增4,672参数，末层零初始化。两臂参数均7,465,988（共享7,431,248／头34,740），从同一v7 full best64加载全部497个旧state，新核3个state同seed初始化；各500个state初始逐tensor一致。

候选Q(a,p,b)=M(p,b)T(a,p,b)exp(v(a))/c_v(p,b)，其中M=sum_aT、c_v=sum_aTexp(v)；匹配global为Texp(v)/Z。两臂只在参考更新的PB边缘守恒上不同。完整公式与边界见[结构设计](GEOPRR_V2_REFERENCE_PAIR_PRESERVING_DESIGN_CN.md)。M与c参与梯度，不能detach、删状态或做硬参考路由。

原三项loss仍为精确三角色joint CE/(3log256)+fieldL1+progressCDF，权重各1，从epoch1有效；CE按Q的正规化精确计算，没有新增监督或新loss。推理只有RGB ROI，H／span／y与角色相位仅用于训练和计分，虚拟P不是物理针尖，经验query不是针体。

两臂同完整best64权重和新核初始化、同fresh AdamW／scheduler、数据、增强、标签、执行批和曝光。实际有效更新与AMPskip分别记录，不承诺两臂完全相等。与best48／best64比较包含fresh optimizer和继续训练曝光；只有同信息global完成后才检验机制收益。

## 正式安排与原数据边界

- 新输出：`runs/geoprr_v2_tail/iteration_v8_reference_pair_update`，候选preserve先运行，global仅在完整候选通过下述三组gate后运行。
- 每臂65轮，前5轮冻结共享权重和统计，第6轮共同finetune；train batch32、val16、ROI768／输出192。
- backbone lr1e−4、head lr5e−4、cosine minlr1e−6、weight decay.01、gradient clip5、骨干AMP／头与读数／loss FP32／原标签FP64；同原训练合同。
- 原14494／1506／4000划分不动；沿用已冻结source训练18590 unique、22686 donor＋8192 recipient＝30878图像前向／709执行批每轮。每臂65轮应有46085执行批、2007070训练图像前向和97890验证图像；有效update／AMPskip按实际记录。
- 完整1506源开发集的原生NMAE最小、严格改善时选best，平局保留最早epoch。不中途以严重图表现换best，不逐图拼接，不以反事实代替训练对照。
- 不运行officialtest、新seed、旧separable／local／product或额外矩阵；不把已查看源开发称盲测、显著性或真实冻结迁移。

## 原11项G1：保持原定义

既有G0标签量化raw max≤1%沿用原已过结果，不重跑教师。其余10项RGB检查继续使用既有compare_to_ordinary，均在完整1506的最终best上执行：

| 检查 | 原冻结门槛，%FS |
|---|---|
| 有效人口 | 完整1506全部有效，失败不得移除分母 |
| NMAE | ≤固定r8的0.1299504985 |
| worst151 mean | ≤0.7205782515 |
| 最大误差 | 严格小于19.5571362972 |
| 旧严重6 | 全部≤2% |
| 新增错误 | 相对普通native和固定r8，均无新增>1／2／5 |
| 共同普通1495 mean | ≤0.0972051518 |
| 共同普通1495 median | ≤0.0564150512 |
| 共同普通1495 p90 | ≤0.2246701717 |
| r8普通1497 mean | ≤0.0977505576 |

实现读取保存原预测精确比较，不以表格四舍五入数替代。

## 两组额外保护：分别报告，不能改写原11项

每组6项：全1506 NMAE、共同普通1495 mean／median／p90、r8普通1497 mean均不差相应anchor，并且相对该anchor无新增>2／5错误。

| Anchor | 全1506 NMAE | 普通1495 mean／median／p90 | r8普通1497 mean |
|---|---:|---|---:|
| 保留v6 feedback best48 | 0.0636129929167 | 0.0581475116493／0.0321567058563／0.121930837631 | 0.0584579313363 |
| 保留v7 full best64 | 0.0591462310626 | 0.0546682457361／0.0309824943542／0.116738080978 | 0.0572620347470 |

只在**G0＋原10RGB＋best48全部6项＋best64全部6项**均通过后，controller才允许global正式训练。任一失败保存完整负结果、成本和阈值ID，跳过global，不降低门槛或补旧控制。

## 匹配完成后的G2与交付

候选相对global的NMAE至少降低5%，worst151和>5数量不增加，且候选的原G1与两组anchor保护通过，才支持单seed源开发机制筛选。普通样本、严重／救回／新增及全部成本仍单独报告；不由通过G2宣称统计显著性或论文创新认证。

两臂实际完成后或候选gate失败后，收齐protocol／config／run／history／best／completed及完整1506逐图。比较ordinary native／固定r8／best48／best64；重点3172、2810、3663、9323同时报告，另报告最大误差、>1／2／5、救回／新增IDs、旧严重6、普通1495／1497、分位／worst151和Acc。真实成本包含全部执行批、有效update、AMPskip、训练／验证图像、完整wall和真实峰值。不能用preflight代替正式成本。

## 准入、进程与归档

新增3项CPU算子与初始化接口PASS；实际RGB资源唯一job192990 PASS，原497旧state与新核500state迁移／两臂初始一致，最大有效batch64非零探针共享梯度有限。2uniqueRGB×两臂共132图像前向、4backward、0optimizer，仅是准入而非训练效果；不重复测试。准入记录见[CPU／RGB资源归档](data/geoprr_v2_reference_pair_update_20261022/README.md)。

启动前source服务器项目env并核对实际进程／GPU空闲；pipeline lock、pipeline.pid、queue_state与当前step原日志共同判定，不重复活任务，DataLoader workers不是新模型。best不更新不代表停滞。真实失败先读原日志与活命令，修真实实现／环境问题并必要针对检查，同output恢复；保留partial与负结果，不删覆盖，实质方法／config／名单改变另开实验。

只使用项目venv和已有main，不建分支，不更改系统Python／CUDA／conda或其他项目、不买算力、不向第三方发消息。RGB／权重／vendor／.aris／private trace／原H／C／XYZ／像素及矩阵不公开；run.json／data_contract含原几何只private，公开使用显式allowlist。所有旧模型保留。

## 论文目标的前瞻性续跑修订

用户已在当前候选完成前明确以数据改进与方法贡献支持论文为主要目标。原G1／anchor／G2分数和判定保留，不再用零新增错误阻断必要机制对照；当前65轮与两臂配置不变，续跑规则见[论文证据修订](GEOPRR_V2_PAPER_EVIDENCE_AMENDMENT_CN.md)。原controller不在live期间改写；如其按原gate退出，matched global按单独修订记录继续，禁止重复活job。
