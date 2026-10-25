# Image structure source control

状态：新实际source零更新AMP预检通过，固定30E训练进行中，终点native/条件CF尚未评估。科学d6b2db442a1d89799dff4024cd6baa26c0f2d28c已冻结推existingmain。

固定RGB-derived梯度结构测量在reader内，原source scalar/完整MNV4/meanSmoothL1保持。模型learned参数8435793/all233，onefixedkernel新增一state为465，cachedencoder462值保留不等feature语义保留。Source-native与条件RGswap都须过联合guard，trueI/R/source03为0RGB/model。Onefixedfilter是额外F32compute非free，event/counter/fullwall/peak记录；不GTmask/roles/TTA/多encoder。详[协议](../../GEOPRR_V2_IMAGE_STRUCTURE_SOURCE_CONTROL_CN.md)、[机器协议](protocol.json)、[启动](launch.json)、[实际source witness](runtime_preflight.json)。只explicitallowlist metadata公开，rawassets留private。
