import numpy as np
from ultralytics import YOLO
import cv2
import math
import os
import ultralytics
ultralytics.checks()

# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)

# Load a model
# model = YOLO('runs/detect/train2/weights/last.pt')  # load a partially trained model

# Resume training
# results = model.train(resume=True)

model = YOLO('weights/yolov8n.pt')  # load a pretrained YOLOv8n detection model
# results = model("./output/pic.jpg", show=True, save=True)

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

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

cap.set(3, 1920)
cap.set(4, 1080)

output_folder = './output'
output_index = 1

while os.path.exists(os.path.join(output_folder, f'output{output_index}.mov')):
    output_index += 1

output_file = os.path.join(output_folder, f'output{output_index}.mov')


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(f'../output/{output_file}', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    success, img = cap.read()
    results = model(source=img, stream=True, conf=0.60, device='mps')

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class
            cls = int(box.cls[0])

            # DisplayText
            cv2.putText(img, f'{conf}: {classNames[cls]}', [x1, y1 - 20], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Write the frame into the output video file
    out.write(img)

    cv2.imshow('TEST', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

out.release()

cv2.destroyAllWindows()
