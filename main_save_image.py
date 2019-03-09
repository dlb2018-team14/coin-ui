from darkflow.net.build import TFNet
import cv2
import os

# darkflowのオプション
# オプションの一覧はこちら
# https://github.com/thtrieu/darkflow/blob/b2aee0000cd2a956b9f1de6dbfef94d53158b7d8/darkflow/defaults.py#L8
options = {
        # 訓練済みモデル
        "load": "./data/weights/yolo-obj_900.weights",
        # 訓練するときに使ったyoloのコンフィグファイル
        "model": "./data/cfg/yolo-obj.cfg",
        # ラベル
        "labels": "./data/labels/obj.names",
        # 閾値
        "threshold": 0.3,
        # 予測対象の画像ディレクトリ
        "imgdir": "./data/sample-images/"
}

# sample-images/outディレクトリを作っておかないと、静かに落ちる
os.makedirs("./data/sample-images/out")

tfnet = TFNet(options)

#imgcv = cv2.imread("./data/sample-images/IMG_0411_frame_1.jpg")
#result = tfnet.return_predict(imgcv)
#print(result)
tfnet.predict()
