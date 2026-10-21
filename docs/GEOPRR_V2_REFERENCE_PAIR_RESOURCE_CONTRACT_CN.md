# 配对守恒参考更新：有界RGB接口与资源合同

日期：2026-10-22。只验证新Reader实际接口、数值与资源，0 optimizer，不启动正式训练。既有CPU算子／初始化检查不重复，不重跑完整1506诊断或旧GT教师。

## 固定预算与输入

- 唯一原初始化：v7 full best64、seed20262020，原497个state逐tensor加载，新核3个state；两臂初始全相同。保留原checkpoint，不写权重。
- 唯一2张实际RGB：原验证manifest中sync_3172、sync_9323，原ROI／768／192和原数据配置。仅这2条监督合同与保存预测核对；不是新名单或完整集评估。
- 两臂串行：pair_preserving、matched global；每臂一次batch2冻结共享warmup前向＋反传，随后一次最大有效batch64共享finetune前向＋反传。64仅由同2图重复，不称64独立图。
- 总预算4次CNN batch调用、132张CNN图像前向、132原state-head与132新verifier图像调用、4次loss反传、0 optimizer、0实际训练update。readout及新归约调用实际单独计数，继承base前向的原J／readout计算全部计入。
- entry为experiments.preflight_geoprr_v2_reference_pair_update；唯一输出diagnostics/reference_pair_update_preflight_20261022_v1。output／pid／progress／completed保护防重复。执行前source experiments/geoprr_v2_env.sh并核实实际GPU／进程为空。

## 两种必要工作负载

1. 冻结共享权重和统计，新核v=0：两图实际RGB读数重放best64≤1e−5；新旧角色CE／fieldL1／CDF／total≤1e−5，新修正为0。验证CSV接口保留ID／group／target与完整概率。反传后旧共享无梯度／统计变化，新核末层梯度有限且非零。
2. 共享finetune压力batch64：只在该临时内存模型上，把新末层设为固定 `(index−15.5)×1e−3/32`，两臂相同、与标签／val读数无关，确保GPU执行非零v路径。该探针不是训练超参或新候选权重，不保存；原497权重仍从同best64开始。骨干AMP，头／读出／loss FP32，标签原FP64。所有trainable梯度存在且有限，允许初始化或数据导致部分梯度为0；全质量误差≤1e−4。不得将这次非零探针当作模型精度。

读取原三项loss，对联合角色CE使用精确Q正规化；M与c不得detach。检查新末层实际梯度、共享finitebackward与完整读数接口。没有Adam或其状态，报告真实峰值allocated／reserved、阶段forward+loss／完整forward+backward以及从模块imports起的进程wall；不拿此结果代替正式训练峰值或速度，不比较两臂先后缓存时序作算法加速结论。

## 失败与后续

若原生重放／loss／概率／梯度不符，停止并保留原日志、partial与真实阶段计数，仅修真实新实现问题后做必要针对检查；不得删输出覆盖。若OOM，不自动降低batch、tile、曝光或删状态绕过本合同，先核对实际进程和失败栈再修精确实现。无异常即归档本次资源结果。

通过本合同也不启动正式训练。用户当前要求先不训练、不加loss、不扩模型seed；原11门槛与best48／best64保护不变。RGB、权重、原H／C／XYZ和矩阵仅private，公开只标量、结构配置与计数allowlist。
