import math
from ultralytics import YOLO
import cv2

# load a pretrained model (recommended for training)
model = YOLO('weights/yolov8n.pt')

# class_name = ['jeepney code']
classNames = ["human", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

img = cv2.imread('./output/pic.jpg')

results = model(img, save=True, project='./output')[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, confidence, cls = result
    confidence = math.ceil(confidence * 100) / 100
    x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)

    w, h = x2 - x1, y2 - y1

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f"{classNames[cls]} : {confidence}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

cv2.imshow("Test", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
