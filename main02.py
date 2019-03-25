import cv2
import numpy as np

# yolo v3対応版 webカメラ

# 参考：
# https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py
# https://github.com/sankit1/cv-tricks.com/blob/master/OpenCV/Running_YOLO/predict_on_yolo.py
# https://nixeneko.hatenablog.com/entry/2018/08/15/000000

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


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        labelName = LABELS[classId]
        label = '%s %.2f' % (labelName, conf)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])

        cv2.rectangle(
            frame,
            (left, top - labelSize[1]), # (left, top)
            (left + labelSize[0], top + baseLine), #(right, bottom)
            (255, 255, 255), # 色
            cv2.FILLED #  thickness: 線の太さ
        )

        cv2.putText(
            frame,
            label,
            (left, top),
            cv2.FONT_HERSHEY_SIMPLEX, # フォント
            0.5, # フォントサイズ
            (0, 0, 0), # 色
            #5 # thickness: 線の太さ
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
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

cap = cv2.VideoCapture(0)

# https://stackoverflow.com/questions/31821451/opencv-resizing-window-does-not-work
#winName = 'Deep learning object detection in OpenCV'
#cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
winName = ''
cv2.namedWindow(winName, 0)
_, frame = cap.read()
height, width, _ = frame.shape
print(height, width)
cv2.resizeWindow('', width, height)

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
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
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv2.imshow(winName, frame)
