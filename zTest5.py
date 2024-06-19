import math
import cv2
from ultralytics import YOLO
from util import read_jeepcode
import os


# load model
jeepcode_model = YOLO('weights/jeep300.pt')

# load image
img = cv2.imread('output/images/jeepney (31).jpg')  # Specify the path to your image

className = ["jeep code"]
classNo = [0]
# detect jeepney
results = jeepcode_model(img, verbose=False)[0]

for detection in results.boxes.data.tolist():
    x1, y1, x2, y2, confidence, classId = detection

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    classId = math.ceil(classId)
    confidence = math.ceil(confidence * 100) / 100

    if classId in classNo:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, f'{className[classId]} : {confidence}', (x1, y1), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 2)

        crop_jeepney_code = img[y1:y2, x1:x2, :]
        crop_jeepney_code_gray = cv2.cvtColor(crop_jeepney_code, cv2.COLOR_BGR2GRAY)

        jeepneycode_text, jeepneycode_score = read_jeepcode(crop_jeepney_code_gray)

        # Apply additional filters and OCR attempts if initial read is None
        if jeepneycode_text is None:
            # Filter 0: Thresholding
            _, crop_jeepney_code_thresh = cv2.threshold(crop_jeepney_code_gray, 64, 255, cv2.THRESH_BINARY)
            jeepneycode_text, jeepneycode_score = read_jeepcode(crop_jeepney_code_thresh)

            if jeepneycode_text is None:
                # Filter 1: Adaptive Thresholding
                crop_jeepney_code_adaptive_thresh = cv2.adaptiveThreshold(crop_jeepney_code_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                jeepneycode_text, jeepneycode_score = read_jeepcode(crop_jeepney_code_adaptive_thresh)

                if jeepneycode_text is None:
                    # Filter 2: Morphological Transformations
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    crop_jeepney_code_morph = cv2.morphologyEx(crop_jeepney_code_gray, cv2.MORPH_OPEN, kernel)
                    jeepneycode_text, jeepneycode_score = read_jeepcode(crop_jeepney_code_morph)

                    if jeepneycode_text is None:
                        # Filter 3: Canny Edge Detection
                        crop_jeepney_code_edges = cv2.Canny(crop_jeepney_code_gray, 100, 200)
                        jeepneycode_text, jeepneycode_score = read_jeepcode(crop_jeepney_code_edges)

        if jeepneycode_text is not None:
            cv2.putText(img, f'{jeepneycode_text} : {math.ceil(jeepneycode_score * 100) / 100}%', (100, 50), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 255, 0), 3)

# Save the processed image

file_path = './jeepney.jpg'
base, extension = os.path.splitext(file_path)
counter = 1

while os.path.exists(file_path):
    file_path = f"{base}_{counter}{extension}"
    counter += 1

cv2.imwrite(file_path, img)  # Specify the path to save folder
