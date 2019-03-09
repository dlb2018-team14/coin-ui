from darkflow.net.build import TFNet
import cv2

options = {
    # webcam
    # https://github.com/thtrieu/darkflow#cameravideo-file-demo
    "demo": "camera",
    # 訓練済みモデル
    "load": "./data/weights/yolo-obj_900.weights",
    # 訓練するときに使ったyoloのコンフィグファイル
    "model": "./data/cfg/yolo-obj.cfg",
    # ラベル
    "labels": "./data/labels/obj.names",
    # 閾値
    "threshold": 0.3
}

tfnet = TFNet(options)

tfnet.camera()
