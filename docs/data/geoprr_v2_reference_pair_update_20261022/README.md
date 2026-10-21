# 配对守恒参考更新：CPU实现准入

精确算子、匹配global算子和RGB Reader已实现。新增CPU算子检查及best64初始化／缓存接口检查PASS；没有实际RGB前向、GPU调用或optimizer更新，没有新训练结果。

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

下一步先形成有界的实际RGB接口／资源合同，再检查当前GPU和真实进程；本阶段没有执行该检查。当前没有训练入口、正式队列或新的1506模型评估。不能把CPU缓存重放当成真实RGB重放、batch64共享finetune内存、AMP稳定或训练性能证明。

Reader继承旧full前向，会先计算原生J／读数再作新更新；未来实际成本需要包括这部分计算，不能只报新算子速度。原11门槛、best48／best64保护、冻结人口与模型seed不变；当前继续遵守不训练、不加loss、不扩seed。

结构公式、匹配global对照、原CE及研究边界见[设计合同](../../GEOPRR_V2_REFERENCE_PAIR_PRESERVING_DESIGN_CN.md)。一般KL投影／局部MLP不作为新定理，新机制收益与创新尚未由匹配训练验证。
