from darkflow.net.build import TFNet
import cv2

options = {
        # 訓練済みモデル
        "load": "./data/weights/maeharin_yolo-obj_900.weights",
        # 訓練するときに使ったyoloのコンフィグファイル
        "model": "./data/cfg/maeharin_yolo-obj.cfg",
        # ラベル
        "labels": "./data/labels/obj.names",
        # 閾値
        "threshold": 0.3
}

tfnet = TFNet(options)

# 予測対象の画像
imgcv = cv2.imread("./data/sample-images/IMG_0411_frame_1.jpg")
result = tfnet.return_predict(imgcv)
print(result)
tfnet.predict(imgcv)
