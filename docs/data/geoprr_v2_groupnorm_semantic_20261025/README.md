# V31：单图分组归一化语义读出

**真实源AMP零更新见证已通过，正式源训练已启动；尚无完整源epoch或外域结果。** [启动核实](launch_verification.json)记录2026-10-25T09:17:58.154572+00:00时唯一GPU source child387513、13900MiB，pipeline387472；首批epoch1 batch1／64图，loss0.21511368453502655、LR0.0003、gradient norm finite。Industrial／RF100尚未启动，不能据此给出泛化结论。

[协议](../../GEOPRR_V2_GROUPNORM_SEMANTIC_ZERO_SHOT_PROTOCOL_CN.md)固定完整缓存MobileNetV4内77个BN替换为8组GroupNorm，复制原affine并保留原drop→act；原231个learned cached encoder参数tensor逐项保持，原231个BN运行buffer明确移除。实际8435793参数／233个trainable参数tensor／233state、GN77／BN0／buffer0；encoder为8434512参数／231state，GAPsemantic1280与新Linear1280→1 point，全参数源训练。保留learned参数不保证原预训练函数不变。

GN使用固定8组／eps1e-5，train与eval均在当前单张图内部按固定公式计算分组channel及空间统计，不发生batch耦合、running-stat累计、目标人口统计拟合或eval参数适配。77个GN的feature／affine／归一化数学均在关闭autocast后以FP32执行，再转回输入feature dtype并经过原drop→act；外层encoder／semantic／point为AMP FP16，scalar sigmoid为FP32。eval确实计算单图统计，不能写成不计算任何目标输入统计。

[保存runtime见证](runtime_preflight.json)为3CNN／8图／1combinedback／0optimizer；实际输入RGB512／output128，semantic1280及scalar路径、全部233参数在0optimizer下不变、233 strict reload、encoderstem／完整neck／GN affine／point必要梯度有限非零，train／eval均无model buffer。该记录只支持本次执行见证，不能替代完整优化或外域评估。

12项新增CPU检查已PASS，不重复执行。源人口、ROI／scalar标签／RGB512／原photo、SmoothL1、四项mean-tail源R及min60max160合同保持；本目录[显式allowlist](public_allowlist.json)共7项，只公开规范化配置及必要metadata／shape／dtype／bool。probe sampleID名单、实际state列表、完整population、commands、rawrun／fullcontracts、RGB／features／瞬时统计值／weights／geometry及privatepaths保持私有。
