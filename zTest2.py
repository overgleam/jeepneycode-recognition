from ultralytics import YOLO

import ultralytics
import easyocr

ultralytics.checks()

# from sort.sort import *

# Initialize the OCR reader
reader = easyocr.Reader(['en'])
detections = reader.readtext('./output/william.png')
print(detections)