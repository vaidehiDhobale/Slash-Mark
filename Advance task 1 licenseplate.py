# Import necessary libraries
import cv2
import numpy as np
from sort.sort import Sort
from jetson.utils import loadImageRGBA
import jetson.inference
import jetson.utils

# Initialize SORT tracker
mot_tracker = Sort()

# Load YOLOv4 models
coco_model = jetson.inference.detectNet("coco/detectnet_coco_resnet18_640x480_ppm", threshold=0.5)
license_plate_detector = jetson.inference.detectNet("license_plate/license_plate_detector", threshold=0.5)

# Load video (replace with your video file path)
video_path = './sample.mp4'
cap = cv2.VideoCapture(video_path)

# List of vehicle class IDs (adjust as needed)
vehicle_classes = [2, 3, 5, 7]

# Process frames
frame_nmr = -1
results = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_nmr += 1
    results[frame_nmr] = {}

    # Detect vehicles using COCO-trained YOLOv4
    cuda_frame = jetson.utils.cudaFromNumpy(frame)
    detections = coco_model.Detect(cuda_frame, frame.shape[1], frame.shape[0])

    detections_ = []
    for detection in detections:
        if detection.ClassID in vehicle_classes:
            detections_.append([detection.Left, detection.Top, detection.Right, detection.Bottom, detection.Confidence])

    # Track vehicles using SORT
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    cuda_frame = jetson.utils.cudaFromNumpy(frame)
    license_plates = license_plate_detector.Detect(cuda_frame, frame.shape[1], frame.shape[0])

    for license_plate in license_plates:
        if license_plate.ClassID == 0:  # Assuming license plate class ID is 0
            x1 = license_plate.Left
            y1 = license_plate.Top
            x2 = license_plate.Right
            y2 = license_plate.Bottom
            score = license_plate.Confidence

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process license plate (convert to grayscale, threshold, etc.)
                # Implement your read_license_plate function here

                # Store results
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': license_plate_text}

# Save or display results as needed
# Implement your
cap.release()
cv2.destroyAllWindows()
