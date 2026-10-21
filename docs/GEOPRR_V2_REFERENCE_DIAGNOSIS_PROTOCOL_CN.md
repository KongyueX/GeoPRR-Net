# best57有限角色／配对诊断合同

缓存阶段已完成：sync3172/338/1999/12540恰为完整1506的reference entropy最高4张，也为joint-product stateL1最高4张。缓存反事实对3张joint剩余严重例改善，但sync1999恶化到16.960302%FS。该关联不证明根因，不能据源验证设entropy路由门槛或逐图GT选读数。

## 固定执行

只使用v5 joint best57、seed20262020和原1506源验证顺序、batch16、原768/192预处理、骨干AMP／头FP32。每张RGB只做一次骨干前向，0训练／0optimizer／0新seed。不运行被gate跳过的product训练。先查真实GPU与进程，使用独立diagnostics输出目录并保留partial，禁止重复活任务。

第一检查必须重放正式保存的1506 joint／同checkpoint product读数，ID/group/target/valid相同，最大归一化读数差≤1e-5。超过门槛先查预处理／AMP／checkpoint，不覆盖正式成绩。干预完整保留质量与人口，输入失败error1。

## 两个坐标分支

1. Native：原冻结头输出，用于重放及按参考的依赖贡献分解。不把teacher角度标签在错误预测坐标下的替换解释成纯角色因果。
2. Teacher-rectifier：重用同一批已生成F4/context，只额外执行冻结的采样／关系头，center/SPD换成已有训练标签。记录1506次native头＋1506次teacher标定头调用，骨干仍仅1506图像前向。teacher H/center/SPD及角色只用于条件诊断，不能成为部署输入。teacher重采样存在分布变化，不证明训练因果。

只有在teacher坐标分支上，固定做A、P、B、P+B、A+P、A+B和A+P+B七项替换。A+P／A+B用于区分参考和单角色同时错误，A+P+B仅为已有标签量化的算术一致性control；不作新的精度发现。P/B的绝对twohot按每个参考a转相对索引；不只替换预测argmax参考，不按GT选择干预读数。所有替换先统一保存，再计分完整1506和固定cohort。

## 最小保存

全1506、每坐标分支保存A[256]、delta/span边缘[256]、joint/product progress PMF[256]；每参考条件读数r_a[256]、P/B entropy、teacher邻接bin质量、零跨度／弧外低高端质量。保存有符号贡献c_a=A_a(E[P_aB_a]f-E[P_a Bbar]f)，要求sum c_a=r_joint-r_product且sum A_a*r_a=r_joint，数值容差预定1e-5。f严格沿用原展开／除span／clip／span0uniform映射；不能换成两均值比或普通协方差。

不保存全人口J／完整attention／FPN／RGB。只为预先固定sync3172/12540/338/1999保存完整P_rel/B_rel，供峰和交叉状态重放。teacher标定和原始坐标只private。替换读出保存完整progress PMF、读数、质量摘要。实际CNN／额外头／reader调用和wall／峰值分别计数。

## 会改变下一动作的结果

- teacher标定重采样显著改善固定错误并保住完整普通指标：优先改坐标／采样结构；若仍坏，不继续笼统归因四参数标定。
- teacher坐标后teacher-A进一步改善且真实起点行的条件读数好：优先检验角色证据反向约束参考的交互；不只增加自由参考混合分量。
- teacher-P或B单轴替换在teacher坐标下改善：聚焦对应局部角色内容；只有P+B联合才改善则记录交互，不能唯一归因单头。
- 高A参考下P/B多峰的交叉组合承载错误进度质量：进一步检验同一a内成对P-B证据／读数空间结构；仅称与条件独立交叉错误相容，不能凭一张GT断言真实配对身份。

各干预报告完整1506、共同普通1495、r8普通1497、旧严重6、全部新增错误、NMAE/Acc/分位/tail/max及成本；GT条件结果不是新RGB模型。机制可能共存，不强迫每图唯一根因。执行代码必须先对小PMF算术和坐标分支做有意义的针对检查，再执行一次正式诊断；不重跑已通过的旧几何／环境测试。
