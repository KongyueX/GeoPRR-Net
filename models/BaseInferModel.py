# models/BaseInferModel.py
import threading
from abc import ABC, abstractmethod
from typing import Any


class BaseInferModel(ABC):
    """
    单例模式的基础类，用于实现单例模式。
    子类需要实现以下方法:
    - load_model: 加载模型。
    - preprocess: 预处理。
    - infer: 模型推理。
    - postprocess: 后处理。
    """
    _instances = {}  # 改为字典存储各子类的单例实例
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> 'BaseInferModel':
        if cls not in cls._instances:
            with cls._lock:  # 线程安全
                if cls not in cls._instances:  # 双重检查锁定
                    instance = super().__new__(cls)
                    # 仅在初始化成功后缓存实例，避免半初始化对象污染后续请求
                    instance.init(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    def init(self, *args, **kwargs) -> None:
        """
        初始化方法，用于加载模型和设置其他必要的参数。
        需要实现的方法:
        - load_model: 加载模型。
        - preprocess: 预处理。
        - infer: 模型推理。
        - postprocess: 后处理。
        - infer_withCapturing: 模型视频流持续推理。

        Args:
            model_name (str): 模型名称。
            model_path (str): 模型路径。
        """
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        单例模式的初始化方法，子类调用时会自动继承此文档。

        Note:
            这是一个空方法，实际初始化逻辑在init()方法中。
            子类可以通过super().__init__()调用此方法。
        """
        # 不要在__init__中执行任何逻辑，因为单例模式下__init__可能会被多次调用
        pass

    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        """预处理"""
        pass

    @abstractmethod
    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """模型推理"""
        pass

    @abstractmethod
    def infer_withCapturing(self, *args: Any, **kwargs: Any) -> Any:
        """模型视频流持续推理"""
        pass

    @abstractmethod
    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        """后处理"""
        pass
