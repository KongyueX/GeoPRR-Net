from loguru import logger
import base64
import cv2
from models.BaseInferModel import BaseInferModel

class MyInferModel(BaseInferModel):
    """
    供接口联调使用的最小模型模板，不参与指针表生产推理或论文实验。

    新模型需要实现以下方法:
    - load_model: 加载模型。
    - preprocess: 预处理。
    - infer: 模型推理。
    - postprocess: 后处理。
    """
    def __init__(self):
        super().__init__()

    def init(self):
        self.isCapture = False
        self.model = None  # 模型对象，用于存储模型实例
        self.load_model()  # 加载模型
    
    @logger.catch(reraise=True)
    def load_model(self):
        logger.info("loading model")
        self.model = "This is a model"
        logger.info("model loaded")

    def infer(self, *args, **kwargs):
        """
        模型推理总线。
        Args:
            args: 位置参数。
            kwargs: 关键字参数。
        Returns:
            result (json): 模型推理的结果。
        """
        if self.model is None:
            self.load_model()
        image_data = kwargs.get('image', None)
        config = kwargs.get('config', None)
        if image_data is None:
            return {"status": False, "message": "image_data is None", "result": []}
        image_data = self.preprocess(imageData=image_data, config=config)
        self.infer_image_data(image_data)
        _, buffer = cv2.imencode(".jpg", image_data)
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        return {"status": True,
                "message": "infer success",
                "result": "infer result",
                # 保留历史拼写，避免破坏已有联调客户端。
                "preprcoess": image_base64,
                }

    def infer_image_data(self, image):
        logger.info("infer finished")
        return image
    
    def preprocess(self, *args, **kwargs):
        """模板默认不修改输入图像。"""
        imageData = kwargs.get('imageData', None)
        logger.info("preprocess finished")
        return imageData

    def postprocess(self):
        """模板默认不执行后处理。"""
        pass

    def infer_withCapturing(self, *args, **kwargs):
        return self.infer(*args, **kwargs)
