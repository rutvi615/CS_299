import cv2

cap = cv2.VideoCapture('Re-Experiment[2]\MVI_4599.MP4')
cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    output_path = f"twoframe.jpg"
    cv2.imwrite(output_path, frame)
    break