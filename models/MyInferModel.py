from loguru import logger
import base64
import cv2
from models.BaseInferModel import BaseInferModel

class MyInferModel(BaseInferModel):
    """
    继承BaseInferModel的子类，用于实现具体的模型推理逻辑。
    这是一个模板，可以用来参考
    子类需要实现以下方法:
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
    
    @logger.catch(reraise=True)  # 捕获异常并记录日志，reraise=True表示继续抛出异常，否则会被catch捕获，导致程序退出，这里可以根据需要设置为False，也可以不设置，默认是False，即不抛出异常，程序会继续执行，不会退出，但是会记录日志，方便调试
    def load_model(self):
        # 子类实现具体的模型加载逻辑
        logger.info("loading model")
        # TODO: 加载模型
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
        # 先检测模型是否加载成功
        if self.model is None:
            self.load_model()
        # 前处理
        image_data = kwargs.get('image', None)  # 获取输入图像数据，根据实际情况修改参数名和类型
        config = kwargs.get('config', None)  # 获取输入图像数据，根据实际情况修改参数名和类型
        if image_data is None:
            return {"status": False, "message": "image_data is None", "result": []}
        image_data = self.preprocess(imageData=image_data, config=config)  # 调用子类实现的前处理方法，返回处理后的图像数据，或者其他需要的参数
        # 模型推理
        result = self.infer_image_data(image_data)  # 调用子类实现的模型推理方法，返回模型推理的结果，或者其他需要的参数
        # 返回检测结果
        _, buffer = cv2.imencode(".jpg", image_data)  # 转换为字节流
        # 进行 base64 编码
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        return {"status": True,
                "message": "infer success",
                "result": "infer result",
                "preprcoess": image_base64,  # TODO: 这里需要返回前处理的结果，用于调试，实际使用时可以删除这行代码，或者根据需要修改返回的结果，比如返回图片的base64编码，或者返回图片的路径，或者返回图片的numpy数组，或者返回图片的tensor，或者返回图片的shape，或者返回图片的dtype，或者返回图片的nu
                }

    def infer_image_data(self, image):
        # 子类实现模型推理逻辑
        logger.info("infer finished")
        return image
        # inferData = kwargs.get('inferData', None)
        # dataType = kwargs.get('dataType', None)
    
    def preprocess(self, *args, **kwargs):
        # 子类实现预处理逻辑
        imageData = kwargs.get('imageData', None)
        config = kwargs.get('config', None)
        # TODO: 实现模型前处理逻辑
        logger.info("preprocess finished")

        return imageData  # 返回处理后的图像数据，或者其他需要的参数

    def postprocess(self):
        # 子类实现后处理逻辑
        pass

    def infer_withCapturing(self, *args, **kwargs):
        return self.infer(*args, **kwargs)
