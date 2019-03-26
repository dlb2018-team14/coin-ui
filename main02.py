import cv2
import datetime
import numpy as np

# yolo v3対応版 webカメラ

#
# opencvでyolov3
# https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py
# https://github.com/sankit1/cv-tricks.com/blob/master/OpenCV/Running_YOLO/predict_on_yolo.py
# https://nixeneko.hatenablog.com/entry/2018/08/15/000000
#
# opencvのカメラの使い方
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_gui/py_video_display/py_video_display.html
#

# kato yolov2
#VIDEO_NAME = "./video-kato-yolov2-threshold0.5-{0}.mp4".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
#MODEL = "./data/weights/kato_yolo-obj_900.weights"
#CFG = "./data/cfg/kato_yolo-obj.cfg"
# fukase yolov2
#VIDEO_NAME = "./video-fukase-yolov2-{0}.mp4".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
#MODEL = "./data/weights/fukase_yolo-obj.weights"
#CFG = "./data/cfg/fukase_yolo-obj.cfg"
# abe yolov3
VIDEO_NAME = "./video-abe-yolov3-{0}.mp4".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
MODEL = "./data/weights/abe_yolov3_6000.weights"
CFG = "./data/cfg/abe_yolov3.cfg"
SCALE = 0.00392 # 1/255, 入力のスケール
INP_SHAPE = (416, 416) #入力サイズ
MEAN = 0
RGB = True

net = cv2.dnn.readNetFromDarknet(CFG, MODEL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confThreshold = 0.3 # Confidence threshold
nmsThreshold = 0.4  # Non-maximum supression threshold

LABELS = [
    "1yen",
    "5yen",
    "10yen",
    "50yen",
    "100yen",
    "500yen",
    "other",
]

# rgb
# https://www.rapidtables.com/web/color/RGB_Color.html
COLORS = [
    (0, 0, 0),        # 1yen:black
    (128,128,0),      # 5yen:Olive
    (128, 0, 0),      # 10yen:maroon
    (0, 0, 255),      # 50yen:blue
    (0, 255, 0),      # 100yen:lime
    (255, 0, 0),      # 500yen:red
    (128, 128, 128),  # other:grey
]


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # color
        # rgb -> bgr
        classColor = COLORS[classId]
        classColor = (classColor[2], classColor[1], classColor[0])

        # label
        label = LABELS[classId]

        # 座標
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        # Draw a bounding box.
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            classColor
        )

        fontType = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 1
        fontThickNess = 2

        labelSize, baseLine = cv2.getTextSize(
            label,
            fontType,
            fontSize,
            fontThickNess
        )

        top = max(top, labelSize[1])

        cv2.rectangle(
            frame,
            (left, top - labelSize[1]), # (left, top)
            (left + labelSize[0], top + baseLine), #(right, bottom)
            classColor, # 色
            cv2.FILLED #  thickness: 線の太さ
        )

        cv2.putText(
            frame,
            label,
            (left, top),
            fontType, # フォント
            fontSize, # フォントサイズ
            (255, 255, 255), # 色(白）
            fontThickNess # thickness: 線の太さ
        )

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []

    if lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = center_x - width / 2
                    top = center_y - height / 2
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    labels = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        classId = classIds[i]
        drawPred(classId, confidences[i], left, top, left + width, top + height)
        labels.append(LABELS[classId])

    # 総額計算
    labels = [l.replace('yen', '') for l in labels]
    nums = [int(l.replace('yen', '')) for l in labels if l != 'other']
    amountText = "%d yen" % sum(nums)

    fontType = cv2.FONT_HERSHEY_SIMPLEX
    fontSize = 2
    fontThickNess = 5
    labelSize, baseLine = cv2.getTextSize(
        amountText,
        fontType,
        fontSize,
        fontThickNess
    )
    left = 0
    top = 50
    cv2.rectangle(
        frame,
        (left, top - labelSize[1]), # (left, top)
        (left + labelSize[0], top + baseLine), #(right, bottom)
        (0, 0, 200), # 色
        cv2.FILLED #  thickness: 線の太さ
    )
    cv2.putText(
        frame,  # 書き込み対象画像
        amountText,  # 書き込みテキスト
        (left, top),  # org: 書く場所の座標(テキストを書き始める位置の左下)
        fontType,  # fontFace: フォント ( OpenCVが提供するフォントの情報については cv2.putText() 関数のドキュメンテーションを参照のこと)-
        fontSize,  # fontScale: フォントサイズ (文字のサイズ)
        (255, 255, 255),  # color
        fontThickNess,  # thickness: 線の太さ
        # False  # bottomLeftOrigin(第9引数): Trueなら左下隅を原点、そうでなければ左上隅
    )


def capture_camera(camera, videoWriter, winName):
    while camera.isOpened():
        hasFrame, frame = camera.read()
        if not hasFrame:
            cv2.waitKey()
            break

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Create a 4D blob from a frame.
        inpWidth = INP_SHAPE[0]
        inpHeight = INP_SHAPE[1]
        blob = cv2.dnn.blobFromImage(frame, SCALE, (inpWidth, inpHeight), MEAN, RGB, crop=False)

        # Run a model
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        postprocess(frame, outs)

        # Put efficiency information.
        #t, _ = net.getPerfProfile()
        #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        #cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv2.imshow(winName, frame)
        videoWriter.write(frame)

        # any key pressed, then cancel
        # https://theasciicode.com.ar/ascii-control-characters/escape-ascii-code-27.html
        choice = cv2.waitKey(1)
        if choice > 0:
            print("canceled!")
            break


if __name__ == "__main__":

    # カメラ設定
    camera = cv2.VideoCapture(0)
    # https://stackoverflow.com/questions/31821451/opencv-resizing-window-does-not-work
    #winName = 'Deep learning object detection in OpenCV'
    #cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    winName = ''
    cv2.namedWindow(winName, 0)
    _, frame = camera.read()
    height, width, _ = frame.shape
    cv2.resizeWindow('', width, height)

    # 録画設定
    # 動画コーデックを指定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 録画のフレームレート指定
    # 本当は一回処理を走らせて計測したほうがいいけど
    fps = 3
    videoWriter = cv2.VideoWriter(VIDEO_NAME, fourcc, fps, (width, height))

    try:
        capture_camera(camera, videoWriter, winName)
    finally:
        print("finalize...")
        videoWriter.release()
        camera.release()
        cv2.destroyAllWindows()

