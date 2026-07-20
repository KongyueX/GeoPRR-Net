import os
import sys
import logging
from types import FrameType
from typing import cast
from loguru import logger
# from .path_conf import LogPath
 
 
class Logger:
    """输出日志到文件和控制台"""
 
    def __init__(self):
        folder_ = "./log/user_log/"
        # folder_ = logpath
        prefix_ = "PipeStatusBackend-"
        rotation_ = "00:00"
        # rotation_ = "10 MB"
        retention_ = "30 days"
        encoding_ = "utf-8"
        backtrace_ = True
        diagnose_ = True
        enqueue_ = True

        # 格式里面添加了process和thread记录，方便查看多进程和线程程序
        format_ = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> ' \
                    '| <magenta>{process}</magenta>:<yellow>{thread}</yellow> ' \
                    '| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>'

        # 这里面采用了层次式的日志记录方式，就是低级日志文件会记录比他高的所有级别日志，这样可以做到低等级日志最丰富，高级别日志更少更关键
        # debug
        self.logger = logger
        # 清空所有设置
        # self.logger.remove()
        # 判断日志文件夹是否存在，不存则创建
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        self.logger.add(folder_ + prefix_ + "debug.log", level="DEBUG", backtrace=backtrace_, diagnose=diagnose_,
                    format=format_, colorize=False, enqueue=enqueue_,
                    rotation=rotation_, retention=retention_, encoding=encoding_,
                    filter=lambda record: record["level"].no >= logger.level("DEBUG").no)

        # info
        self.logger.add(folder_ + prefix_ + "info.log", level="INFO", backtrace=backtrace_, diagnose=diagnose_,
                    format=format_, colorize=False, enqueue=enqueue_,
                    rotation=rotation_, retention=retention_, encoding=encoding_,
                    filter=lambda record: record["level"].no >= logger.level("INFO").no)

        # warning
        self.logger.add(folder_ + prefix_ + "warning.log", level="WARNING", backtrace=backtrace_, diagnose=diagnose_,
                    format=format_, colorize=False, enqueue=enqueue_,
                    rotation=rotation_, retention=retention_, encoding=encoding_,
                    filter=lambda record: record["level"].no >= logger.level("WARNING").no)

        # error
        self.logger.add(folder_ + prefix_ + "error.log", level="ERROR", backtrace=backtrace_, diagnose=diagnose_,
                    format=format_, colorize=False, enqueue=enqueue_,
                    rotation=rotation_, retention=retention_, encoding=encoding_,
                    filter=lambda record: record["level"].no >= logger.level("ERROR").no)

        # critical
        self.logger.add(folder_ + prefix_ + "critical.log", level="CRITICAL", backtrace=backtrace_, diagnose=diagnose_,
                    format=format_, colorize=False, enqueue=enqueue_,
                    rotation=rotation_, retention=retention_, encoding=encoding_,
                    filter=lambda record: record["level"].no >= logger.level("CRITICAL").no)

        self.logger.add(sys.stderr, level="CRITICAL", backtrace=backtrace_, diagnose=diagnose_,
                    format=format_, colorize=True, enqueue=enqueue_,
                    filter=lambda record: record["level"].no >= logger.level("CRITICAL").no)




        # # 文件的命名
        # log_name = f"Fast_{time.strftime('%Y-%m-%d', time.localtime()).replace('-', '_')}.log"
        # log_path = os.path.join(LogPath, "Fast_{time:YYYY-MM-DD}.log")
        # self.logger = logger
        # # 清空所有设置
        # self.logger.remove()
        # # 判断日志文件夹是否存在，不存则创建
        # if not os.path.exists(LogPath):
        #     os.makedirs(LogPath)
        # # 日志输出格式
        # formatter = "{time:YYYY-MM-DD HH:mm:ss} | {level}: {message}"
        # # 添加控制台输出的格式,sys.stdout为输出到屏幕;关于这些配置还需要自定义请移步官网查看相关参数说明
        # self.logger.add(sys.stdout,
        #                 format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
        #                        "{process.name} | "  # 进程名
        #                        "{thread.name} | "  # 进程名
        #                        "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
        #                        ":<cyan>{line}</cyan> | "  # 行号
        #                        "<level>{level}</level>: "  # 等级
        #                        "<level>{message}</level>",  # 日志内容
        #                 )
        # # 日志写入文件
        # self.logger.add(log_path,  # 写入目录指定文件
        #                 format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
        #                        "{process.name} | "  # 进程名
        #                        "{thread.name} | "  # 进程名
        #                        '{module}.{function}:{line} - {level} -{message}',  # 模块名.方法名:行号
        #                 encoding='utf-8',
        #                 retention='7 days',  # 设置历史保留时长
        #                 backtrace=True,  # 回溯
        #                 diagnose=True,  # 诊断
        #                 enqueue=True,  # 异步写入
        #                 rotation="00:00",  # 每日更新时间
        #                 # rotation="5kb",  # 切割，设置文件大小，rotation="12:00"，rotation="1 week"
        #                 # filter="my_module"  # 过滤模块
        #                 # compression="zip"   # 文件压缩
        #                 )
 
    def init_config(self):
        LOGGER_NAMES = ("uvicorn.asgi", "uvicorn.access", "uvicorn")
 
        # change handler for default uvicorn logger
        logging.getLogger().handlers = [InterceptHandler()]
        for logger_name in LOGGER_NAMES:
            logging_logger = logging.getLogger(logger_name)
            logging_logger.handlers = [InterceptHandler()]
 
    def get_logger(self):
        return self.logger
 
 
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
 
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1
 
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )
 
 
Loggers = Logger()
log = Loggers.get_logger()