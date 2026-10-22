# 配对守恒参考更新：CPU实现准入

精确算子、匹配global算子和RGB Reader已实现。新增CPU检查及有界实际RGB／GPU资源准入均PASS；CPU／资源准入阶段没有optimizer更新；正式候选已按最新授权启动，正式训练成本和结果单独记录。

## 实现与检查

- `experiments/geoprr_v2_reference_pair_update.py`：绝对PB边缘在log域按A分块累积，第二pass精确运输到相对J；两pass均checkpoint反传，M／c不detach，不存完整K³。
- `experiments/geoprr_v2_reference_pair_reader.py`：已有9方向token的共享4,672参数内容核、best64全state迁移、真实更新后A／J／读数、原三项监督的精确CE正规化。匹配global臂使用相同核与输入。
- `experiments/test_geoprr_v2_reference_pair_update.py`：只新增3项算子检查，不重跑旧组件或全1506诊断。

| 新CPU检查 | 结果 |
|---|---|
| tinyK完整枚举／绝对PB守恒／全质量／真实A／相对J | PASS |
| 精确联合CE、全部5个API输入梯度及v0基础梯度一致 | PASS |
| 循环roll／base能量常数shift／tile／集中有限logits | PASS |
| 两臂原best64的497个state精确迁移 | PASS |
| 各500个state／7,465,988参数／两臂初始逐tensor相同 | PASS |
| 3172／9323缓存能量＋合成token，v0读数重放 | 最大差3.576278687×10⁻7 |
| v0完整概率质量／CE修正 | 最大差5.960464478×10⁻7／0 |
| 零末层初始化的新核首步梯度 | 有限且末层非零 |

3项算子检查共18次operator调用、8次CE helper、10次梯度查询；另有新核首步1次梯度查询。Reader接口2次更新、4次verifier图像等价调用、4次readout模块调用／8图像等价。**这些是合成或缓存组件调用，全部新增CNN图像前向／GPU调用／optimizer更新均为0。**

算子进程1.768055838s（unittest部分0.794s）；Reader合同3.580099151s从imports后开始计时，不是完整进程wall。执行只使用服务器项目venv，GPU在CPU脚本中不可见。记录分别见[算子CPU检查](operator_cpu_checks.json)和[Reader CPU检查](reader_cpu_checks.json)。

## 下一步与边界

有界实际RGB／资源检查已按[冻结合同](../../GEOPRR_V2_REFERENCE_PAIR_RESOURCE_CONTRACT_CN.md)完成，执行前已核对GPU／实际进程为空；结果见下节。用户已明确授权候选65轮及通过原11／best48／best64保护后的匹配global65；[正式合同](../../GEOPRR_V2_REFERENCE_PAIR_UPDATE_PROTOCOL_CN.md)、两份配置和训练／控制器入口已冻结，启动状态单独记录。没有新的已完成训练成绩。

Reader继承旧full前向，会先计算原生J／读数再作新更新；未来实际成本需要包括这部分计算，不能只报新算子速度。原11门槛、best48／best64保护、冻结人口与模型seed不变；正式候选按最新授权运行，继续不加loss、不扩seed。

结构公式、匹配global对照、原CE及研究边界见[设计合同](../../GEOPRR_V2_REFERENCE_PAIR_PRESERVING_DESIGN_CN.md)。一般KL投影／局部MLP不作为新定理，新机制收益与创新尚未由匹配训练验证。

## 实际RGB／GPU资源准入

唯一job192990已正常退出。仅原3172／9323两张RGB，分别用于两臂冻结v0 batch2和重复图像压力batch64，共4次骨干调用／132张CNN图像前向、132原头／132新verifier图像调用、4次新更新、24次readout模块调用／792图像等价、4次完成反传、0optimizer。没有重复完整1506。

| 阶段 | 配对守恒臂 | matched global臂 |
|---|---:|---:|
| v0实际RGB重放best64最大差 | 8.642673492×10⁻7 | 8.940696716×10⁻7 |
| v0原三项objective重放最大差 | 0 | 0 |
| v0新末层梯度绝对最大值 | 0.026050590 | 0.025983954 |
| 压力有效batch／实际uniqueRGB | 64／2 | 64／2 |
| 固定非零探针evidence绝对最大值 | 0.002611599 | 0.002611599 |
| 压力概率质量最大误差 | 7.748603821×10⁻7 | 2.384185791×10⁻7 |
| 压力实际peak allocated bytes | 23,662,420,992 | 23,645,643,776 |
| 压力实际peak reserved bytes | 29,267,853,312 | 29,267,853,312 |
| 共享finetune梯度 | 全部存在且有限 | 全部存在且有限 |

两臂初始全部500state相同，原best64的497state完整保留。冻结阶段共享权重／BN统计不变；压力阶段的BN与新末层探针仅存在于临时模型，没有保存权重或作optimizer更新。固定探针 `(index−15.5)×1e−3/32` 与标签／验证读数无关，只确保非零v路径实际执行。

从模块imports开始的完整进程wall22.989027020s；实际阶段时序全部保存在[原始标量报告](runtime_preflight.json)。两臂先后执行的编译／缓存状态不同，阶段时序不用于算法速度对比。骨干使用AMP、头／读出／loss FP32；本检查使用未缩放loss.backward，没有GradScaler或Adam状态，不能替代正式训练的AMPskip、有效更新、速度或峰值记录。重复2张图的batch64也不能证明64张不同图的训练行为。

实际RGB与资源准入通过不代表原11项精度门槛通过，也没有新模型性能或算法因果结论。正式训练按用户最新明确授权执行候选65轮，global仅候选通过原11及两组anchor保护后运行；不加loss、不扩seed。

[独立资源标量审计](runtime_independent_audit.json)PASS：完成记录与process.log最终记录一致，实际调用／预算／初始化／零势重放／非零探针和成本统计一致；审计只读本地标量，没有重跑模型。

## 正式队列合同

新队列iteration_v8_reference_pair_update，preserve候选先65轮，过三组gate后才global65；两臂同完整best64初始化及原曝光／优化安排。[新比较器与配置标量接线检查](formal_metadata_checks.json)PASS：best64自比6／6，已知best64相对best48新增3172负例正确拒绝（5／6），0新CNN／GPU／optimizer；这是门槛接线检查，不是新模型效果。正式成本与完整1506逐图结果待各臂实际完成后归档。

## 正式启动快照

2026-10-22北京时间04:29:20启动preserve候选65轮，训练代码d6376a0。父193689／子193692实际命令核对一致，GPU只有该候选；global未启动。首批epoch1／head_warmup实际32donor＋11recipient，loss0.0901810601有限；启动核验时完整history为0轮，这不是后续实时进度或方法结论。实际最新进度以服务器history／latest和当前step日志为准。

[启动标量记录](formal_launch_state.json)保留同best64初始化、曝光、三组gate和进程；全部权重／原始run.json只在server／private归档。保留v6best48、v7best64及旧失败结果。候选完整65轮通过原11＋best48／best64保护后才matched global65；否则归档负结果并跳过global。

[北京时间2026-10-22 10:01当前best2逐图比较](interim_best2_20261022_1001.md)：完成38轮、正在39，非最终65轮门槛；不覆盖正式结果。

## 论文证据目标

用户在候选完成前明确：严格鲁棒门槛不是论文的一票否决条件，优先建立数据改进与方法机制的匹配证据。[前瞻性续跑修订](../../GEOPRR_V2_PAPER_EVIDENCE_AMENDMENT_CN.md)保留原gate结果及当前65轮配置，仅允许必要的已冻结global对照在原严格gate未全过时按独立记录续跑；不把原失败改成通过。
