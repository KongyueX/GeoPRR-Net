# camera_process_manager.py
import cv2
import multiprocessing
import queue
import time
from typing import Dict, Optional
from loguru import logger

class CameraProcessService:
    """
    相机进程管理器，用于创建和管理多个相机读取进程
    """

    def __init__(self):
        # 使用管理器创建可以在进程间共享的字典
        self.manager = multiprocessing.Manager()
        self.is_capturing_shared = self.manager.dict()

        # 存储各个相机相关信息的字典
        self.camera_processes: Dict[str, multiprocessing.Process] = {}
        self.camera_queues: Dict[str, multiprocessing.Queue] = {}
        self.cameras: Dict[str, cv2.VideoCapture] = {}

    def start_camera_capture(self, camera_id: str, camera_type: Optional[str] = None) -> tuple[bool, str]:
        """
        启动指定相机ID的捕获进程

        Args:
            camera_id: 相机ID
            camera_type: 相机类型（如"usb_dshow"等）

        Returns:
            tuple: (success, message)
        """
        camera_id = str(camera_id)

        # 检查是否已经存在该相机的进程
        if camera_id in self.camera_processes and self.camera_processes[camera_id].is_alive():
            return False, f"Camera {camera_id} is already running"

        # 创建队列用于传输图像数据
        self.camera_queues[camera_id] = multiprocessing.Queue(maxsize=3)

        # 在共享字典中设置状态
        self.is_capturing_shared[camera_id] = True

        # 创建并启动进程
        process = multiprocessing.Process(
            target=self._camera_capture_process,
            args=(camera_id, camera_type, self.camera_queues[camera_id], self.is_capturing_shared)
        )
        process.daemon = True
        process.start()

        # 保存进程引用
        self.camera_processes[camera_id] = process

        return True, f"Started camera {camera_id} capture process"
    # services/CameraProcessService.py
    def stop_camera_capture(self, camera_id: str, timeout: int = 10) -> tuple[bool, str]:
        """
        停止指定相机ID的捕获进程

        Args:
            camera_id: 相机ID
            timeout: 超时时间（秒）

        Returns:
            tuple: (success, message)
        """
        camera_id = str(camera_id)

        # 检查相机是否存在
        if camera_id not in self.camera_processes.keys():
            return False, f"Camera {camera_id} is not running"

        # 更新共享状态以通知进程停止
        self.is_capturing_shared[camera_id] = False

        # time.sleep(3)

        # 等待进程结束
        process = self.camera_processes[camera_id]

        # 先尝试非阻塞检查进程是否已经结束
        if process is None or not process.is_alive():
            # 进程已经结束，清理资源
            if camera_id in self.camera_queues:
                del self.camera_queues[camera_id]

            if camera_id in self.camera_processes:
                del self.camera_processes[camera_id]

            if camera_id in self.is_capturing_shared:
                del self.is_capturing_shared[camera_id]

            logger.info(f"Camera {camera_id} capture process already stopped")
            return True, f"Stopped camera {camera_id} capture process"

        # 进程仍在运行，等待其结束
        process.join(timeout=timeout)

        if process.is_alive():
            # 如果进程仍未结束，强制终止
            logger.info(f'强制终止相机{camera_id}')
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                # 如果仍然无法终止，可能需要kill
                try:
                    process.kill()
                    process.join(timeout=1)
                except AttributeError:
                    # Python < 3.7 没有kill方法
                    pass
            return False, f"Camera {camera_id} process termination timeout"

        # 清理资源
        if camera_id in self.camera_queues:
            del self.camera_queues[camera_id]

        if camera_id in self.camera_processes:
            del self.camera_processes[camera_id]

        if camera_id in self.is_capturing_shared:
            del self.is_capturing_shared[camera_id]

        logger.info(f'相机{camera_id}已停止')
        return True, f"Stopped camera {camera_id} capture process"

    def get_image(self, camera_id: str, timeout: float = 1.0) -> tuple[bool, object]:
        """
        从指定相机获取图像

        Args:
            camera_id: 相机ID
            timeout: 获取图像的超时时间（秒）

        Returns:
            tuple: (success, image or error message)
        """
        camera_id = str(camera_id)

        if camera_id not in self.camera_queues:
            return False, f"Camera {camera_id} queue not found"

        try:
            image = self.camera_queues[camera_id].get(timeout=timeout)
            return True, image
        except queue.Empty:
            return False, "No image available"

    def list_cameras(self) -> list:
        """
        获取正在运行的相机列表

        Returns:
            list: 正在运行的相机ID列表
        """
        return [cam_id for cam_id, process in self.camera_processes.items() if process.is_alive()]

    def stop_all_cameras(self, timeout: int = 10):
        """
        停止所有相机捕获进程

        Args:
            timeout: 每个进程的超时时间（秒）
        """
        camera_ids = list(self.camera_processes.keys())
        for camera_id in camera_ids:
            self.stop_camera_capture(camera_id, timeout)
    # services/CameraProcessService.py
    @staticmethod
    def _camera_capture_process(camera_id: str, camera_type: Optional[str],
                               image_queue: multiprocessing.Queue,
                               is_capturing_shared: dict):
        """
        相机捕获进程的实际执行函数

        Args:
            camera_id: 相机ID
            camera_type: 相机类型
            image_queue: 图像队列
            is_capturing_shared: 共享的捕获状态字典
        """
        # 打开相机
        if camera_id.isdigit():
            camera_capture_id = int(camera_id)
        else:
            camera_capture_id = camera_id

        logger.info(f"Opening camera {camera_id}")

        if camera_type == "usb_dshow":
            cap = cv2.VideoCapture(camera_capture_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_capture_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return

        logger.info(f"Camera {camera_id} opened successfully")

        count = 0
        max_retry_count = 125

        # 持续捕获图像直到收到停止信号
        try:
            while is_capturing_shared.get(camera_id, False):
                # 检查相机连接状态，必要时重连
                if not cap.isOpened() or count >= max_retry_count:
                    logger.warning(f"Reconnecting camera {camera_id}")
                    cap.release()
                    time.sleep(1)
                    if camera_type == "usb_dshow":
                        cap = cv2.VideoCapture(camera_capture_id, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(camera_capture_id)
                    count = 0
                    continue

                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    count += 1
                    time.sleep(0.01)  # 短暂休眠避免过度占用CPU
                    continue

                count = 0  # 重置计数器

                # 将图像放入队列
                try:
                    image_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        # 如果队列满，移除旧帧并添加新帧
                        image_queue.get_nowait()
                        image_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

        except KeyboardInterrupt:
            logger.info(f"Camera {camera_id} capture process interrupted")
        except Exception as e:
            logger.error(f"Camera {camera_id} capture process error: {e}")
        finally:
            # 释放资源
            try:
                cap.release()
            except:
                pass
            logger.info(f"Camera {camera_id} capture process stopped")
            while not image_queue.empty():
                image_queue.get()
            return

# 使用示例
if __name__ == "__main__":
    # 创建相机管理器实例
    manager = CameraProcessService()

    try:
        # 启动多个相机（示例使用0和1作为相机ID）
        success, message = manager.start_camera_capture("0")
        print(f"Starting camera 0: {success}, {message}")

        success, message = manager.start_camera_capture("1", "usb_dshow")
        print(f"Starting camera 1: {success}, {message}")

        # 等待一段时间让相机启动
        time.sleep(2)

        # 显示正在运行的相机
        print(f"Running cameras: {manager.list_cameras()}")

        # 获取图像示例（这里只是演示，实际使用时可能需要在循环中处理）
        success, image = manager.get_image("0", timeout=1.0)
        if success:
            print(f"Got image from camera 0, shape: {image.shape}")
        else:
            print(f"Failed to get image from camera 0: {image}")

        # 运行一段时间
        time.sleep(10)

        # 停止特定相机
        success, message = manager.stop_camera_capture("0")
        print(f"Stopping camera 0: {success}, {message}")

        # 停止所有剩余相机
        manager.stop_all_cameras()
        print("All cameras stopped")

    except KeyboardInterrupt:
        print("Interrupted by user, stopping all cameras...")
        manager.stop_all_cameras()
    except Exception as e:
        print(f"Error occurred: {e}")
        manager.stop_all_cameras()
