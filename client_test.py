import argparse
import cv2
import numpy as np
import requests
import base64
import os

from loguru import logger

# 配置服务地址（与main.py中配置的host和port一致）
BASE_URL = "http://127.0.0.1:30600"

def test_root_endpoint():
    """测试根路径/"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, "根路径请求失败"
    assert response.json() == {"message": "Hello World"}, "根路径返回内容错误"
    print("根路径测试通过")

def test_items_endpoint():
    """测试/items/{item_id}"""
    test_item_id = 123
    response = requests.get(f"{BASE_URL}/items/{test_item_id}")
    assert response.status_code == 200, "items路径请求失败"
    assert response.json() == {"item_id": test_item_id}, "items路径返回内容错误"
    print("items路径测试通过")

def test_infer_test_endpoint():
    """测试/infer_test（需要model_list.yaml中存在MyInferModel）"""
    model_name = "MyInferModel"  # 对应model_list.yaml中的key
    test_infer_data = "test_data"
    response = requests.get(f"{BASE_URL}/infer_test/{model_name}/{test_infer_data}")
    
    assert response.status_code == 200, "infer_test路径请求失败"
    if "status" in response.json() and not response.json()["status"]:
        print(f"infer_test警告：{response.json()['message']}（可能是模型加载问题，请检查model_list.yaml和模型类）")
    else:
        print("infer_test路径基本测试通过")

def test_infer_local_file_endpoint(test_image_path):
    """测试/infer_local_file（需要model_list.yaml中存在MyInferModel）"""
    # 测试filepath类型（需准备本地测试图片）
    if not os.path.exists(test_image_path):
        print(f"警告：{test_image_path}不存在，跳过filepath类型测试")
    else:
        payload = {
            "modelName": "PointerMeterInferModel",
            "inferData": test_image_path,
            "dataType": "filepath",
            "cameraTimeout": 10,
            "inferConfig": {"device": "0",
                            "result_pointer_image": True,
                            "result_mask_image": True,
                            # "scaleEnd": 1.0,
                            # "confidence": 0.5,
                            }
        }
        logger.info(f"开始测试(file_path){test_image_path}")
        response = requests.post(f"{BASE_URL}/infer", json=payload)
        assert response.status_code == 200, "infer(filepath)请求失败"
        # 只打印关键信息
        response_json = response.json()
        logger.info(f"[{os.path.basename(test_image_path)}] infer(filepath)测试通过: "
              f"status={response_json.get('status')}, "
              f"message={response_json.get('message')}, "
              f"检测结果={response_json.get('result')}")
        # logger.info("infer(filepath)基本测试通过")

def test_infer_base64_endpoint(test_image_path):
    # 测试base64类型（使用test_image.jpg生成base64）
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as f:
            base64_str = base64.b64encode(f.read()).decode()
        payload = {
            "modelName": "PointerMeterInferModel",
            "inferData": base64_str,
            "dataType": "base64",
            "cameraTimeout": 10,
            "inferConfig": {"device": "0",
                            "result_pointer_image": True,
                            "result_mask_image": True,
                            # "confidence": 0.5,
                            }
        }
        logger.info(f"开始测试(base64){test_image_path}")
        response = requests.post(f"{BASE_URL}/infer", json=payload)
        assert response.status_code == 200, "infer(base64)请求失败"

        # 只打印关键信息
        response_json = response.json()
        logger.info(f"[{os.path.basename(test_image_path)}] infer(base64)测试通过: "
              f"status={response_json.get('status')}, "
              f"message={response_json.get('message')}, "
              f"检测结果={response_json.get('result')}")
        result_image = response_json.get("result_image")
        result_digital_image = response_json.get("result_digital_image")
        result_pointer_image = response_json.get("result_pointer_image")
        result_mask_image = response_json.get("result_mask_image")
        if result_image:
            result_image = base64.b64decode(result_image)
            nparr = np.frombuffer(result_image, np.uint8)
            result_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("result_image", result_image)
            cv2.waitKey(0)
        if result_digital_image:
            result_digital_image = base64.b64decode(result_digital_image)
            nparr = np.frombuffer(result_digital_image, np.uint8)
            result_digital_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("result_digital_image", result_digital_image)
        if result_pointer_image:
            result_pointer_image = base64.b64decode(result_pointer_image)
            nparr = np.frombuffer(result_pointer_image, np.uint8)
            result_pointer_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("result_pointer_image", result_pointer_image)
        if result_mask_image:
            result_mask_image = base64.b64decode(result_mask_image)
            nparr = np.frombuffer(result_mask_image, np.uint8)
            result_mask_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("result_mask_image", result_mask_image)
        cv2.waitKey(0)
    else:
        logger.info(f"警告：{test_image_path}不存在，跳过base64类型测试")

def test_infer_camera_endpoint():

    # 测试cameraId类型（需有可用摄像头）
    test_camera_id = "0"  # 通常0代表默认摄像头
    payload = {
        "modelName": "FireInferModel",
        "inferData": test_camera_id,
        "dataType": "cameraId",
        "cameraTimeout": 10,
        "inferConfig": {"device": "0"}
    }
    response = requests.post(f"{BASE_URL}/infer", json=payload)
    if response.status_code == 200:
        print("infer(cameraId)基本测试通过")
    else:
        print(f"infer(cameraId)测试失败：{response.json()['message']}（可能是摄像头不可用）")

def test_infer_endpoint(test_image_path=None):
    """测试POST /infer（覆盖三种dataType）"""
    test_image_path = test_image_path or os.environ.get(
        "POINTER_METER_TEST_IMAGE"
    )
    if not test_image_path:
        raise ValueError(
            "provide test_image_path or set POINTER_METER_TEST_IMAGE"
        )
    # test_infer_local_file_endpoint(test_image_path)
    test_infer_base64_endpoint(test_image_path)
    # test_infer_camera_endpoint()
    # if not os.path.exists(test_image_path):
    #     print(f"警告：{test_image_path}不存在，跳过filepath类型测试")
    # else:
    #     payload = {
    #         "modelName": "MyInferModel",
    #         "inferData": test_image_path,
    #         "dataType": "filepath",
    #         "cameraTimeout": 10,
    #         "inferConfig": {"param1": "value1"}
    #     }
    #     response = requests.post(f"{BASE_URL}/infer", json=payload)
    #     assert response.status_code == 200, "infer(filepath)请求失败"
    #     print("infer(filepath)基本测试通过")




def test_endCapture_endpoint():
    """测试POST /endCapture"""
    test_camera_id = "0"
    response = requests.post(f"{BASE_URL}/endCapture", json={"cameraId": test_camera_id})
    assert response.status_code == 200, "endCapture请求失败"
    if response.json()["status"]:
        print("endCapture测试通过")
    else:
        print(f"endCapture测试警告：{response.json()['message']}（可能是未启动捕获）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ad-hoc API integration checks."
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("POINTER_METER_TEST_IMAGE"),
        help=(
            "local image used by /infer; may also be supplied through "
            "POINTER_METER_TEST_IMAGE"
        ),
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="only check lightweight endpoints; do not run model inference",
    )
    args = parser.parse_args()
    test_root_endpoint()
    test_items_endpoint()
    if not args.smoke_only:
        if not args.image:
            parser.error(
                "--image (or POINTER_METER_TEST_IMAGE) is required unless "
                "--smoke-only is used"
            )
        test_infer_endpoint(args.image)
    # test_endCapture_endpoint()
    print("所有测试执行完毕")
