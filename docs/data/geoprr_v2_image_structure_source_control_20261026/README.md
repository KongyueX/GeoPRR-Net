# Image structure source control

状态：root协议固定30E，代码实现中，未启动。唯一新增固定RGB-derived梯度结构测量，原source scalar/完整MNV4/meanSmoothL1不变；source native与条件RGswap联合guard，真实I/R及needle03全0RGB/model。预期0新增trainable、1kernelbuffer、465states，extra filter compute不免费。详[协议](../../GEOPRR_V2_IMAGE_STRUCTURE_SOURCE_CONTROL_CN.md)、[机器协议](protocol.json)。本目录仅显式allowlist metadata，raw资产留private。
