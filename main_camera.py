from darkflow.net.build import TFNet
import cv2
import sys
import os
from time import time as timer
import numpy as np

# see: webcam
# https://github.com/thtrieu/darkflow#cameravideo-file-demo

# オプションの解説：
# https://github.com/thtrieu/darkflow/blob/b2aee0000cd2a956b9f1de6dbfef94d53158b7d8/darkflow/defaults.py#L8
options = {
    # 訓練済みモデル
    #"load": "./data/weights/kato_yolo-obj_900.weights",
    "load": "./data/weights/fukase_yolo-obj.weights",
    # 訓練するときに使ったyoloのコンフィグファイル
    #"model": "./data/cfg/kato_yolo-obj.cfg",
    "model": "./data/cfg/fukase_yolo-obj.cfg",
    # ラベル
    "labels": "./data/labels/obj.names",
    # 閾値
    "threshold": 0.3,
    # 何elapsedごとに結果を画面に表示するか
    "queue": 1,
    # 撮影結果を保存するか
    "saveVideo": True
}

tfnet = TFNet(options)


def start_web_camera(tfnet):
    SaveVideo = tfnet.FLAGS.saveVideo

    # カメラデバイス番号
    device_number = 0

    # webカメラ初期化
    camera = cv2.VideoCapture(device_number)

    # fps設定
    #camera.set(cv2.CAP_PROP_FPS, 10)
    print(camera.get(cv2.CAP_PROP_FPS))

    tfnet.say('Press [ESC] to quit demo')

    assert camera.isOpened(), \
        'Cannot capture source'

    cv2.namedWindow('', 0)
    _, frame = camera.read()
    height, width, _ = frame.shape
    cv2.resizeWindow('', width, height)

    # 動画保存フラグがonの場合は動画レコーダーを初期化
    if SaveVideo:
        # 動画コーデックを指定
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # フレームレート指定
        fps = 1 / tfnet._get_fps(frame)
        if fps < 1:
            fps = 1
        videoWriter = cv2.VideoWriter('result-video.mp4', fourcc, fps, (width, height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()

    elapsed = int()
    start = timer()
    tfnet.say('Press [ESC] to quit demo')
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        # カメラからフレームを読み込む
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break

        # 予測前の処理
        preprocessed = tfnet.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)

        # Only process and imshow when queue is full
        if elapsed % tfnet.FLAGS.queue == 0:
            feed_dict = {tfnet.inp: buffer_pre}
            # 予測！
            net_out = tfnet.sess.run(tfnet.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                # バウンディングボックスを画像にセット
                postprocessed = tfnet.framework.postprocess(single_out, img, False)

                # 総額計算
                boxesInfo = tfnet.return_predict(img)
                labels = [boxInfo["label"].replace('yen', '') for boxInfo in boxesInfo]
                nums = [int(l.replace('yen', '')) for l in labels if l != 'other']
                amount = sum(nums)
                # 金額を書き込む
                # 背景色
                #cv2.rectangle(
                #    postprocessed,
                #    (0, 0),  # (left, top)
                #    (300, 150),  # (right, bottom)
                #    (255, 255, 255),  # color white
                #    cv2.FILLED
                #)
                # テキスト
                cv2.putText(
                    postprocessed,  # 書き込み対象画像
                    "%d yen" % amount,  # 書き込みテキスト
                    (0, 50),  # org: 書く場所の座標(テキストを書き始める位置の左下)
                    0,  # fontFace: フォント ( OpenCVが提供するフォントの情報については cv2.putText() 関数のドキュメンテーションを参照のこと)-
                    2,  # fontScale: フォントサイズ (文字のサイズ)
                    (0, 0, 200),  # color (red)
                    5,  # thickness: 線の太さ
                    # False  # bottomLeftOrigin(第9引数): Trueなら左下隅を原点、そうでなければ左上隅
                )

                # 動画ファイルに保存
                if SaveVideo:
                    videoWriter.write(postprocessed)

                # 結果を画面に表示
                cv2.imshow('', postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()

        choice = cv2.waitKey(1)
        if choice == 27: break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    cv2.destroyAllWindows()


start_web_camera(tfnet)


