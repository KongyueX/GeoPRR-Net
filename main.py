# main.py
import hashlib
import importlib
import os
import queue
import threading
import time
from typing import Annotated, Union, Optional

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import yaml

from config.log import Loggers, log
from models.BaseInferModel import BaseInferModel

# 延迟导入DataProcessService，避免在模块加载时初始化
# from services.DataProcessService import DataProcessService

class ModelListUtil:
    _config_cache = None
    _last_modified = 0
    _lock = threading.Lock()
    _config_file = os.path.join('./models', 'model_list.yaml')
    _current_config = None

    @classmethod
    def get_config(cls):
        return cls._current_config or cls.read_config()

    @classmethod
    def read_config(cls, force_reload=False):
        try:
            with cls._lock:
                current_mtime = os.path.getmtime(
                    cls._config_file) if os.path.exists(cls._config_file) else 0

                if force_reload or not cls._config_cache or current_mtime > cls._last_modified:
                    cls._last_modified = current_mtime

                    with open(cls._config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    cls._config_cache = config
                    cls._current_config = cls._config_cache.copy()

                return cls._config_cache.copy()
        except Exception as e:
            log.error(f"YAML配置加载失败: {str(e)}")
            return cls._config_cache or None

class AppConfigUtil:
    _config_cache = None
    _last_modified = 0
    _lock = threading.Lock()
    _config_file = os.path.join('./config', 'app_config.yaml')
    _current_config = None

    @classmethod
    def get_config(cls):
        return cls._current_config or cls.read_config()

    @classmethod
    def read_config(cls, force_reload=False):
        try:
            with cls._lock:
                current_mtime = os.path.getmtime(
                    cls._config_file) if os.path.exists(cls._config_file) else 0

                if force_reload or not cls._config_cache or current_mtime > cls._last_modified:
                    cls._last_modified = current_mtime

                    with open(cls._config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    cls._config_cache = config
                    cls._current_config = cls._config_cache.copy()

                return cls._config_cache.copy()
        except Exception as e:
            log.error(f"应用配置加载失败: {str(e)}")
            return cls._config_cache or None

class ConfigMonitor(threading.Thread):
    """
    配置监控线程，用于定时检测配置文件是否有更新
    """
    def __init__(self, interval=5):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            prev_md5 = hashlib.md5(
                str(ModelListUtil._current_config).encode()).hexdigest()
            ModelListUtil.read_config(force_reload=False)
            new_md5 = hashlib.md5(
                str(ModelListUtil._current_config).encode()).hexdigest()
            if prev_md5 != new_md5:
                log.info("检测到配置更新，新配置已生效")
            time.sleep(self.interval)

    def stop(self):
        self.running = False

# 使用全局字典缓存已加载模型的单例实例
model_instances = {}
# fastapi_model_instance = None
# 创建一个FastAPI实例
app = FastAPI()
# 挂载静态文件路径
app.mount("/static", StaticFiles(directory="static"), name="static")

# 延迟初始化DataProcessService
def get_data_process_service():
    global dataProcessService
    if 'dataProcessService' not in globals():
        from services.DataProcessService import DataProcessService
        dataProcessService = DataProcessService()
    return dataProcessService

class InferDataModel(BaseModel):
    modelName: str = Field(description="需要调用的模型名称")
    inferData: Union[str, int] = Field(
        description="文件路径或Base64编码或相机ID"
    )
    dataType: str = Field(
        description="传入的推理数据类型",
        json_schema_extra={"enum": ["filepath", "base64", "cameraId"]}
    )
    cameraTimeout: int = Field(10, description="相机超时时间", ge=0, le=1000)
    inferConfig: Optional[dict] = Field(None, description="推理配置")
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    # 获取客户端IP地址和端口号
    client_host = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else "unknown"
    client_info = f"{client_host}:{client_port}"
    log.info(f"收到请求: {request.method} {request.url} 来自 {client_info}")
    # 如果只想记录 POST，可以加判断
    # if request.method == "POST":
    #     log.info(f"收到 POST 请求: {request.url} 来自 {client_info}")

    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    log.info(f"请求处理完成: {request.method} {request.url} 来自 {client_info} - {process_time:.2f}ms")
    return response


# 定义一个根路径的GET请求处理函数
@app.get("/")
@log.catch(reraise=True)
async def root():
    log.info("Hello World")
    return {"message": "Hello World"}

# 定义一个带有路径参数的GET请求处理函数
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    log.info(f"get item_id: {item_id}")
    return {"item_id": item_id}

@app.get("/infer_test/{modelName}/{inferData}")
async def infer_test(modelName: str,inferData: str):
    """
    测试模型推理
    """
    log.info("infer_test")
    model_list_config = ModelListUtil.get_config()
    if model_list_config is None:
        return {"message": "model_list.yaml not found"}
    if modelName not in model_list_config:
        return {"message": "model not found"}
    # 动态加载类
    module_name = f"models.{model_list_config[modelName]['class_file']}"  # 获取模块名
    class_name = model_list_config[modelName]["class_name"]  # 获取类名
    if modelName not in model_instances:
        try:
            module = importlib.import_module(module_name)  # 动态导入模块
            cls = getattr(module, class_name)  # 获取类
            model_instances[modelName] = cls()  # 创建单例并缓存
        except (ModuleNotFoundError, AttributeError) as e:
            log.error(f"加载 {module_name}.{class_name} 失败: {e}")
            return {"status": False, "message": f"加载 {module_name}.{class_name} 失败: {e}"}
    # 调用模型推理
    fastapi_model_instance:BaseInferModel = model_instances[modelName]  # 获取缓存实例
    result = fastapi_model_instance.infer(image="test",dataType='base64')
    return result

@app.post("/infer")
async def infer(inferDataModel: InferDataModel):
    return await run_in_threadpool(sync_infer, inferDataModel)

def sync_infer(inferDataModel: InferDataModel):
    """
    测试模型推理
    """
    log.info("接收到推理请求")
    # 尝试获取图像数据
    dataProcessService = get_data_process_service()
    ret,image = dataProcessService.get_image(imageType=inferDataModel.dataType,
                                             inferData=inferDataModel.inferData,
                                             cameraTimeout=inferDataModel.cameraTimeout,
                                             config=inferDataModel.inferConfig)
    if not ret:
        return {"status": ret, "message": image}

    # 读取模型加载配置文件
    model_list_config = ModelListUtil.get_config()
    if model_list_config is None:
        return {"status": False, "message": "model_list.yaml not found"}
    if inferDataModel.modelName not in model_list_config:
        return {"status": False, "message": "model not found"}
    # 动态加载类
    module_name = f"models.{model_list_config[inferDataModel.modelName]['class_file']}"  # 获取模块名
    class_name = model_list_config[inferDataModel.modelName]["class_name"]  # 获取类名
    # 在/infer路由处理函数中：
    if inferDataModel.modelName not in model_instances:
        try:
            module = importlib.import_module(module_name)  # 动态导入模块
            cls = getattr(module, class_name)  # 获取类
            model_instances[inferDataModel.modelName] = cls()  # 创建单例并缓存
        except (ModuleNotFoundError, AttributeError) as e:
            log.error(f"加载 {module_name}.{class_name} 失败: {e}")
            return {"status": False, "message": f"加载 {module_name}.{class_name} 失败: {e}"}
    # 调用模型推理
    fastapi_model_instance: BaseInferModel = model_instances[inferDataModel.modelName]  # 获取缓存实例
    result = fastapi_model_instance.infer(image=image,config=inferDataModel.inferConfig)
    return result

class beginCaptureWithInfer(threading.Thread):
    """
    边采集图像边推理线程
    """
    def __init__(self, inferData, modelName, cameraTimeout, config:dict) -> None:
        super().__init__()
        self.isCaptureWithInfer = True  # 初始化为True
        self.cameraTimeout = cameraTimeout
        self.inferData = inferData
        self.config = config
        self.resultQueue = queue.Queue(config.get("result_queue_maxsize",3))

        # 初始化模型实例
        model_list_config = ModelListUtil.get_config()
        if not model_list_config:
            raise ValueError("model_list.yaml not found")
        if modelName not in model_list_config:
            raise ValueError(f"Model {modelName} not found in config")

        # 动态加载类
        module_name = f"models.{model_list_config[modelName]['class_file']}"  # 获取模块名
        class_name = model_list_config[modelName]["class_name"]  # 获取类名
        # 在/infer路由处理函数中：
        if modelName not in model_instances:
            try:
                module = importlib.import_module(module_name)  # 动态导入模块
                cls = getattr(module, class_name)  # 获取类
                model_instances[modelName] = cls()  # 创建单例并缓存
            except (ModuleNotFoundError, AttributeError) as e:
                log.error(f"加载 {module_name}.{class_name} 失败: {e}")
                raise RuntimeError(f"加载失败: {e}")
        # 调用模型推理
        self.fastapi_model_instance: BaseInferModel = model_instances[modelName]  # 获取缓存实例

    def run(self) -> None:
        dataProcessService = get_data_process_service()
        log.info("开始边采集图像边推理")
        while self.isCaptureWithInfer:
            ret,image = dataProcessService.get_image(imageType="cameraId",
                                                     inferData=self.inferData,
                                                     cameraTimeout=self.cameraTimeout,
                                                     config=self.config)

            if not ret:
                result = {"status": ret, "message": image}
            else:
                result = self.fastapi_model_instance.infer_withCapturing(image=image,config=self.config)
            try:
                self.resultQueue.put_nowait(result)
            except queue.Full:
                try:
                    self.resultQueue.get_nowait()
                    self.resultQueue.put_nowait(result)
                except queue.Empty:
                    self.resultQueue.put_nowait(result)

    def changeConfig(self, config):
        self.config = config


capture_thread_dict = {}
@app.post("/beginCaptureWithInfer")
async def beginCapture_withinfer(inferDataModel: InferDataModel):
    dataProcessService = get_data_process_service()
    imageType = inferDataModel.dataType
    inferData = inferDataModel.inferData
    cameraTimeout = inferDataModel.cameraTimeout
    config:dict = inferDataModel.inferConfig
    if inferDataModel.inferData is None:
        return {"status": False, "message": "Error: inferData is None"}
    if inferDataModel.dataType is None:
        return {"status": False, "message": "Error: imageType is None"}
    elif imageType != 'cameraId':
        return {"status": False, "message": "Error: imageType is not cameraId"}
    else:
        # 检查摄像头是否正在运行
        if str(inferData) in capture_thread_dict:
            return {"status": False, "message": "Error: cameraId is capturing"}
        try:
            captureThreadHandler = beginCaptureWithInfer(inferData=inferData,
                                                         modelName=inferDataModel.modelName,
                                                         cameraTimeout=cameraTimeout,
                                                         config=config)
        except Exception as e:
            return {"status": False, "message": str(e)}
        captureThreadHandler.daemon = True
        capture_thread_dict[str(inferData)] = captureThreadHandler
        captureThreadHandler.start()
        return {"status": True, "message": "begin capture with infer"}

@app.post("/inferWithCapturing")
async def inferWithCapturing(inferDataModel: InferDataModel):
    """
    获取运行中视频流的检测结果
    Args:
        inferDataModel (InferDataModel): 推理数据模型
    Returns:
        result (json): 推理结果
    """
    imageType = inferDataModel.dataType
    inferData = inferDataModel.inferData
    cameraTimeout = inferDataModel.cameraTimeout
    config = inferDataModel.inferConfig
    if inferDataModel.inferData is None:
        return {"status": False, "message": "Error: inferData is None"}
    if inferDataModel.dataType is None:
        return {"status": False, "message": "Error: imageType is None"}
    elif imageType != 'cameraId':
        return {"status": False, "message": "Error: imageType is not cameraId"}
    else:
        try:
            cameraId = str(inferData)
            if cameraId not in capture_thread_dict:
                return {"status": False, "message": "Error: cameraId is not capturing"}
            if config is not None and config:
                capture_thread_dict[cameraId].changeConfig(config)
            try:
                result = capture_thread_dict[cameraId].resultQueue.get(timeout=cameraTimeout)
                return result
            except queue.Empty:
                return {"status": False, "message": "Error: resultQueue is empty"}
        except Exception as e:
            return {"status": False, "message": str(e)}

# main.py
@app.post("/endCapture")
async def endCapture_cameraId(inferDataModel: InferDataModel):
    """
    接收摄像头ID
    Args:
        cameraId (str): 摄像头ID
    Returns:
        result (json): 相机关闭状态
    """
    dataProcessService = get_data_process_service()
    # 在这里处理接收到的cameraId
    cameraId = inferDataModel.inferData
    if cameraId is None:
        return {"status": False, "message": "Error: cameraId is None"}
    cameraIdStr = str(cameraId)
    if cameraIdStr in capture_thread_dict:
        # 停止线程
        capture_thread_dict[cameraIdStr].isCaptureWithInfer = False
        # 从字典中移除
        del capture_thread_dict[cameraIdStr]
    ret, message = dataProcessService.endCapture(cameraId=cameraId)
    log.info(f"相机id {cameraIdStr} 是否还在字典中：{cameraIdStr in capture_thread_dict.keys()}")
    return {"status": ret, "message": message}


# 添加保护代码，确保在Windows上正确运行
if __name__ == "__main__":
    # 独立配置监控线程
    config_monitor = ConfigMonitor()
    config_monitor.start()

    # 读取应用配置
    app_config = AppConfigUtil.get_config()
    server_config = app_config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 30600)

    config = uvicorn.Config("main:app", host=host, port=port, reload=False)
    server = uvicorn.Server(config)
    # 将uvicorn输出的全部让loguru管理
    Loggers.init_config()
    server.run()

