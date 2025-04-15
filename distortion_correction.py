import cv2
import numpy as np

# Load calibration data
K = np.array([[1543.4757, 0.0, 981.8694],
              [0.0, 1538.5658, 508.0066],
              [0.0, 0.0, 1.0]])
dist_coeff = np.array([0.02686, 0.25143, -0.00033, 0.00847, -0.76526])

# Open original video
cap = cv2.VideoCapture('chessboard2.avi')
if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()
# Output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('chessboard_corrected.avi', fourcc, fps, (frame_width, frame_height))

# Init undistortion map
map1, map2 = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if map1 is None or map2 is None:
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, None, K, (frame_width, frame_height), cv2.CV_32FC1)

    # Undistort
    rectified = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Optional: show before and after
    both = cv2.hconcat([frame, rectified])
    cv2.imshow('Original (Left) vs Corrected (Right)', both)

    out.write(rectified)

    if cv2.waitKey(1) == 27:  # ESC to quit early
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Distortion corrected video saved as 'chessboard_corrected.avi'")