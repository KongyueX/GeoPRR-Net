# 补充实验代码验证

实现覆盖E14独立warm与official_full训练、E20简单gate、E27冻结双视图头、匹配缓存导出、逐专家与分组统计，以及条件触发的C4增强匹配和C13目标域OOF head。

## 已验证

- official_full使用16,000源图与145场景；polar/router共享完整源池，96,000条六条件缓存。warm只选择官方训练池中的共同子集。
- source seed、基础权重、候选运行来源、候选变体及availability策略保持一致；错误链在训练/评测前拒绝。
- 当前loss和EMA函数直接复用，控制组固定预算，不通过评测选择模型。
- C13保持1,395图、52组、六条件完整清单；外层测试不参与训练或epoch选择；标准化仅在训练组拟合。
- 各评分保留失败分母，单模型与等权集成分别计算，配对NMAE/Acc@5采用共同组抽样。
- C4 raw输入与现有条件生成路径逐像素一致，spawn多worker与单worker一致。
- 30+30学习率重启、tiny fixture真实优化、checkpoint保存重载及既有公开API回归检查通过。

最终检查环境为macOS CPU、Python3.12、PyTorch2.11.0、torchvision0.26.0、NumPy1.26.4、OpenCV4.11.0.86。构造测试数据只用于代码检查，不是论文实验结果。

```text
59 passed, 20 subtests passed
```

运行命令：

```powershell
python -m pytest -q test/test_revision_cache.py test/test_revision_controls.py test/test_revision_full_training.py test/test_revision_diagnostics.py test/test_revision_oof_heads.py test/test_revision_raw_baseline.py test/test_unified_pointer_reader.py test/test_geoprr_public_api.py
```

六个新模块的CLI帮助入口、Python/JSON语法和Git空白检查通过。独立集成审查未发现剩余阻断问题。

## Known Issues

正式训练目标为仓库已有Windows/Python3.11/PyTorch2.11.0+cu128环境。本次未执行GPU训练或Windows PowerShell脚本；PowerShell参数与路径流程已做静态检查。真实数据、各seed权重、warm配置、smooth-max温度、availability调用规则和必要的OOF特征定义仍须在训练电脑通过R0恢复。完整流水线延迟需在目标GPU实际测量，不以缓存head计时替代。Industrial native专项保持暂停。

跨环境命令与输入规范见 `REVISION_EXECUTION_PROMPT_20260922.md`；实际运行进展只记录到 `REVISION_EXPERIMENT_TRACKER_20260922.md`。
