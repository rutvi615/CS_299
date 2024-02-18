import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean

def remove_edges(image, edge_threshold):
    edges = cv2.Canny(image, 150, 200)
    edges = cv2.threshold(edges, edge_threshold, 255, cv2.THRESH_BINARY)[1]
    result = cv2.bitwise_and(image, cv2.bitwise_not(edges))
    return result
 
cap = cv2.VideoCapture('MVI_1852_2.mp4')
cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
crop_size = 200
edge_removal_threshold = 200

fgmask_frames = []
prev_centroid = None
prev_frame_time = None
frame_count=0
t=[]
v=[]
d=[]
while True:
    ret, frame = cap.read()

    if not ret:
        break 
    
    # frame = frame[0:600, 650:1920]
    fgmask = fgbg.apply(frame)

    fgmask_no_edges = remove_edges(fgmask, edge_removal_threshold)

    fgmask_frames.append(fgmask_no_edges)

    cv2.imshow('fg', fgmask_no_edges)

    # hsv_frame = cv2.cvtColor(fgmask_no_edges, cv2.COLOR_BGR2HSV)

    # lower_color = np.array([0, 100, 100])
    # upper_color = np.array([20, 255, 255])

    _, binary_mask = cv2.threshold(fgmask_no_edges, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if largest_contour is not None:
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            frame_count+=1
            if frame_count%10==0:
                if prev_centroid is not None and prev_frame_time is not None:
                    displacement_vector = np.array([cx, cy]) - np.array(prev_centroid)
                    time_difference = time.time() - prev_frame_time
                    # velocity = (((displacement_vector[0])**2+(displacement_vector[1])**2)**(1/2)) / time_difference
                    velocity = displacement_vector / time_difference
                    t.append(time_difference)
                    d.append(displacement_vector[0])
                    # v.append((((displacement_vector[0])**2+(displacement_vector[1])**2)**(1/2)))
                    v.append(displacement_vector[0]/time_difference)

                    print("Velocity:", velocity)

                prev_centroid = np.array([cx, cy])
                prev_frame_time = time.time()
        
        cv2.imshow('Velocity Estimation', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

# print(v)
v_new=[]
t_new=[]
d_new=[]
for i in range(len(v)):
    if v[i]>19 and v[i]<32:
        v_new.append(v[i])
        t_new.append(i)
        d_new.append(d[i])
print(v_new)
print(t_new)
print(d_new)
plt.plot(t_new,v_new)
plt.show()
# print(mean(v))
cap.release()
cv2.destroyAllWindows()
