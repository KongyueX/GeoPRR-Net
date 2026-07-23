# 代码、注释与冗余文件审计（2026-07-23）

本审计区分三类内容：

1. 可以安全清理的生产死代码；
2. 尚有兼容或复现价值、暂不删除的候选文件；
3. 看起来重复、但论文消融和来源签名要求必须保留的实验实现。

## 本轮已安全清理

| 位置 | 问题 | 处理 |
|---|---|---|
| `models/PointerMeterInferModel.py` | 把生产适配器误写成“模板”；残留火灾检测字段、未使用 YOLO import、return 后死代码和 digital-image 注释块 | 改为准确职责说明并删除死代码 |
| `models/PointerMeterInferModel.py` | 把 0–100 的 `endNum` 位置索引误写成角度 | API 顶层提示改为“归一化指针位置”；已签名核心文件中的旧措辞暂不改 |
| `main.py` | 三处重复动态导入/缓存模型，且并发请求没有统一缓存锁 | 抽取 `get_model_instance()` 并加锁，三个调用点复用 |
| `main.py` | `sync_infer` 被误注释为测试推理；存在未使用 import、重复文件头和成段注释代码 | 修正文档并删除无效内容 |
| `services/DataProcessService.py` | 重复文件头、注释掉的 import/RGB 转换和紧贴代码的含混注释 | 删除；明确服务统一使用 OpenCV BGR |
| `models/MyInferModel.py` | 模板用途不清，含巨型异常注释与 return 后死代码 | 标明只用于联调；保留历史响应键 `preprcoess` 以免破坏客户端 |

## 建议归档或删除的文件

下列文件未被运行入口、测试或论文流水线引用。本轮先登记，不直接删除，避免误删用户仍在使用
的手工调试入口。

| 文件 | 大小 | 判断 | 建议 |
|---|---:|---|---|
| `test_request.json` | 149,718 B | 内嵌大段 Base64 的一次性请求；`test_base64.json` 已提供可读的 filepath 示例 | 删除，或移到 Git LFS/私有样例目录 |
| `utils/angleDetect/vitTranforms/result/heatmap.png` | 102,731 B | 无源码引用的调试输出 | 删除 |
| `utils/angleDetect/vitTranforms/result/matrix.png` | 1,312,310 B | 无源码引用的调试输出 | 删除 |
| `exp_ellipse_mask_strategies.py` / `_v2.py` | 17,084 / 17,838 B | 同一探索的两版根目录脚本 | 保留 v2，把 v1 和实验说明移入 `tools/legacy/ellipse/` |
| `exp_ellipse_rectify.py` / `_v2.py` | 9,614 / 8,100 B | 同一探索的两版根目录脚本 | 同上 |
| `verify_with_yolo_center.py` / `_v2.py` | 16,864 / 17,825 B | 同一手工验证的两版脚本 | 同上 |
| `exp_ellipse_yolo_center.py` | 5,874 B | 独立但未接入测试的探索脚本 | 与上述脚本一起归档 |
| `models/MyInferModel.py` 与 `/infer_test` | 小 | 仅服务脚手架/联调使用，不参与生产模型 | 若不再需要 API 教学示例，可连同 `client_test.py` 对应检查一起删除 |

前三项可立即减少约 1.49 MiB 的无引用仓库内容。根目录椭圆实验共约 91 KiB，体积不大，
主要问题是版本关系和运行方式不清，而不是磁盘占用。

## 必须保留的“表面重复”

| 内容 | 保留原因 |
|---|---|
| v1 独立方向头、概率方向头、hard route、quality route-v1、不确定性融合、进度校准路由 | 分别对应论文消融、负面结果和方法演进；删除会让表格无法复算 |
| `Geometry-v1/v2`、mean/quality fusion、无门控残差 | 内部消融基线，不是冗余生产分支 |
| VDN 适配、训练、验证和七条件评测脚本 | 唯一已完成的同协议外部模型复现 |
| `__init__.py` | Python 包边界；即使为空也不按普通空文件处理 |
| `static/` 下 Swagger/ReDoc 资源 | FastAPI 文档页面在离线环境使用 |
| `utils/angleDetect/**/result/*.pt` | 当前生产 API 的默认运行权重 |

## 为什么暂不修改部分旧注释

`experiments/collect_predictions.py` 会把以下核心源码的原始 SHA-256 写入预测缓存签名：

- `utils/angleDetect/zeroShotMeter.py`
- `utils/angleDetect/dataloader.py`
- `utils/angleDetect/detect.py`
- 分割、检测和 transformer 适配源码

其中仍有“归一化旋转角度”“deepseek 说了”“效果很 nice”等不准确或不正式的历史注释。
只改注释也会改变字节级哈希，使已冻结的 16,000/4,000/1,797 张预测及鲁棒性缓存无法通过
来源复核。论文可信度高于注释风格，因此本轮不触碰这些已签名文件。

下一版正式协议应先把“功能源码哈希”和“注释/格式哈希”分开，或在新版本号下统一重建缓存，
再一次性清理这些注释。不能手工改写旧产物里的哈希来绕过验证。

## 建议的后续清理顺序

1. 用户确认不再需要三个 PNG/Base64 请求后删除它们；
2. 把七个根目录椭圆探索脚本归档到带 README 的 legacy 目录；
3. 决定是否保留 `MyInferModel` 教学端点；
4. 新建实验协议 v3，抽取多评测器重复的 JSONL、分组 bootstrap 和指标函数；
5. v3 缓存重建时同步清理已签名核心文件里的历史注释。

不建议在投稿前把所有历史实验“压成一个最终脚本”。论文复现需要保留能证明失败路线、消融
和外部基线来源的最小代码集合；真正应删除的是无引用输出和不可解释的一次性脚本。
