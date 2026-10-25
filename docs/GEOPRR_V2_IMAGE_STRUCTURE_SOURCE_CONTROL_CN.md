# 可观察图像结构测量：固定30轮source控制

唯一问题是：以RGB局部梯度的三项二阶测量作为同一个scalar CNN的输入，能否保留足够读数信息，降低源验证尾部，同时守住均值、最大误差、>5数量和原本准确样本的新大错？这是实际图像计算，不需要把旧虚拟Q或未认证的tip/mask当作可见监督；不是新理论、物理标定、GT几何恢复或正式V32。

## 来源与边界

既有可见物理角色监督未准入；C3与mean-tail训练风险控制均未满足联合条件，已经关闭，不在本控制中调参重试。输入测量是一个不同的表示问题，不以source tail gap作为原因已证。[Di Zenzo1986](https://www.sciencedirect.com/science/article/pii/0734189X86902239)的publisher abstract支持经典multichannel tensor-gradient的来源，未核实此编码或仪表效果；[Geirhos等，ICLR2019](https://arxiv.org/abs/1811.12231)只提供shape/texture依赖的一般动机，不保证本局部测量有利。

## 唯一固定测量

现有输入仍是native512 ROI、原photo后、ImageNet标准化的float32 RGB。先反归一化，在关闭autocast的float32段使用one-pixel reflect padding和固定3×3 Sobel：

`Kx=[[-1,0,1],[-2,0,2],[-1,0,1]]`；`Ky=[[-1,-2,-1],[0,0,0],[1,2,1]]`。

一个groups=3的固定filter产生dxR/dyR/dxG/dyG/dxB/dyB。随后

`Jxx=sum_c(dx_c²), Jyy=sum_c(dy_c²), Jxy=sum_c(dx_c*dy_c)`，

`E=[Jxx/48,Jyy/48,.5+Jxy/96]`。

E按原ImageNet常数normalize后进入原完整MNV4/GAP1280/Linear1，encoder/point恢复原AMP、sigmoid scalar float32。仅fixed CUDA filter用可恢复的cuDNN allow_tf32=False scope以兑现F32要求，保留并恢复其余backend flags；主encoder AMP/backend策略不变，不改系统CUDA。对exact raw[0,1]每Sobel方向位于±4，尺度48/96是固定数学界，允许float32roundoff，不clip/fit/重新缩放，不改变ROI坐标、尺寸、标签或空间采样。

RGB通道置换只重排和式，实数代数不变；CPU检验必须rawperm后重新normalize，直接交换normalized通道是另一变换。此性质不证明style、contrast、lighting、pose、needleheight、visibility、OCR、可读性或真实泛化。输入会失去绝对强度、梯度极性和RGB身份，反射/纹理/noise亦可产生强结构。

固定kernel为单一persistent float32 buffer [6,1,3,3]，54常数、0新增trainable参数；mean/std不新增registered buffers。预期learned parameters仍8435793、all233tensors，states为465而非旧464（encoder462＋point2＋kernel1），实际由新运行确认。保留cachedImageNet权重不等于保留其输出或feature语义。不保留image/gradient/J/features或moments跨样本缓存，counter/time/validitybool仅作成本/执行证据。

## 源学习和预算

全部source/scalar/ROI/photo/seed/优化器保持identity参考：冻结5496train/687parents和292ID/73layouts，meanSmoothL1beta.05，freshcached完整encoder/freshpoint std.001bias0，sourceBN，seed20262020、loaderseed+epoch、batch64/workers4、AdamW3e-4wd1e-4clip5/defaultfreshScaler。没有mean-tail风险、C3训练、teacher/GT/mask/额外loss/labels/weights/seed/data或TTA。

固定30E、last30，前29无val，无scheduler/best/延长。30E为identity15的两倍scheduledNN预算：输入语义显著改变，预先给予有界featureadaptation，不保证收敛，也不声称same-budget或因果准确率优势。源门槛失败不增加epoch或调整kernel/scale/padding/mixture/seed/loss。

last30 native292先过原constant0.5半mean/ceil30tail学习gate，再联合：mean不高于identity15；tail严格更低；max不高；>5≤16；reference native error≤1%的102样本中no new>5。该subset只本source-ID，不冒充旧ordinary1495/1497。全部过才同last30/同292labels/ROI固定RGswap[1,0,2]一次，CFmean/tail/max不差identityCF，>5≤17，其自己reference error≤1%的104样本中no new>5。不能复用native102。精确reference值在机器协议，Invalid fullerror/drift1、有效0/1饱和不failure、不删行。所有Acc1/2/5/median/p90/p95、new/rescue/image/group/drift都保存，drift阈值非本实验acceptance。

只有native＋CF全部joint才IMAGE_STRUCTURE_SOURCE_CONTROL_PASSED，否则NOT_RETAINED。算子不变、finite、no-op检查或source学习gate本身不等joint通过，更不等适合real域。Source03/Industrial1395/RF151全部0RGB/model/back/update；未来正式source后才同source-best两real域各一次，不能target调参或根据病例设计标签。

## 成本与检查

| 范围 | 图像呈现 | Encoder / point / fixed-filter batch calls | backward attempts |
|---|---:|---:|---:|
| train30E | 164880 | 2580 each | 2580 |
| native | 292 | 5 each | 0 |
| 条件CF | 292 | 5 each | 0 |
| native未过总计 | 165172 | 2585 each | 2580 |
| CF完成总计 | 165464 | 2590 each | 2580 |

Pref另3encoder/3point/3filter calls、8uniqueRGB/8presentations、1combinedback、0opt。Fixedfilter是额外FP32卷积、padding和代数，不免费；CUDAevents按Op起止、phase后同步再累计GPUelapsed，完整phasewall/peak另报，不把异步CPU计时当device time或每batch强制sync。Event/测量开销包含在实际成本。Unknown point-only time为null，updates＋normalAMPskip实记，不调scale补齐。

NEWCPU只synthetic patch：explicitSobel/reflect/坐标/符号，constant/horizontal/vertical/diag/impulse、finite/F32/bounds、rawRGB六perms/noRNG、onekernelbuffer/zero参数/operator strictreload/counts、新30E budget/protocol/inputmetadata/resume toys；rtol/atol固定2e-6仅verification，不fit数据。0realMNV4construct/RGB/weights/GPU，不旧operator/gates/selector/GT/teacher/tests/math重跑。

一次实际source GPU AMP/sourceBN零更新witness为固定4seen train＋4nativeID，3encoder/filter/point calls8unique、1back0opt。确认cached462初值、465states/1kernel/8435793params233tensors、operatorF32 finite/bounds/shape、encoderpointF16/scalarF32、mandatorystemneckpointgradfinite非零、0opt参数和kernel不变、77BNtrain+1/eval0、strictreload新模型。Dataset rawRGB input identity应保留，encoder input本来是measurement，不错误要求其等于rawRGB。NativeID验证重复decode同4图计CPUwall；不提前sourceCFmodel或额外real source operatorpermutation，六perms已synthetic测试。formal fresh无pref模型/BN/RNG继承，formal默认Scaler不变。

先freeze新protocol/config/code/必要NEWCPU evidence push existingmain、scoped部署3newcode/3snapshotmetadata并在任何outmutation前检查、actual旧父子gone/GPUidle后唯一pref→fresh30。Ownentrytimer在torch前、fullstate-opt/scaler/RNG/history/totals/operatorcontract严格newprotocol恢复，无latest或模型partial不fresh覆盖、CSV无summary不二forward。Truefault先保存original log/actualcmd/partial/code/config；科学配置不随修复变化。Whole2stageexit0父子goneGPUempty后才ONEfresh非作者stdlib保存数字/执行核对，实际source字段和abs1e-12数字容差，不虚构normalized字段或要求bitwisefloat；规范化explicitallowlist existingmain归档全部正负。

## Limitations

Lossy局部tensor未必偏向物体shape，pose/非共面/标定/可见性/文字与源标签原始派生未新认证。Source-ID与人工CF为既有development布局，一个seed不支持SD/显著性/newblindtest或新理论。旧C3 rawlabel覆盖资格及旧GeoPRR pixel/fullweight链未重认证照实保留，source/单域收益不当overall成功。无source mask/GTrole/模板推理输入；RGB、weights、features、moments、fullcontracts、trainarrays、geometry/privatepaths留server，stdout前allowlist。无新seed/foundationweights/算力购买/system环境改动/其他项目/cleanup/officialtest4000/第三方消息或branch/gitadd-A/其他untracked。
