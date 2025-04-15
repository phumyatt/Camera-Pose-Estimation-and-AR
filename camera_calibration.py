import cv2 as cv
import numpy as np

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    video = cv.VideoCapture(video_file)
    img_select = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, board_pattern)
        if found:
            cv.drawChessboardCorners(frame, board_pattern, corners, found)
            cv.imshow("Chessboard Detection", frame)
            key = cv.waitKey(wait_msec)
            if select_all or key == ord('s'):
                img_select.append(frame.copy())
        else:
            cv.imshow("Frame", frame)
            cv.waitKey(wait_msec)
    video.release()
    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize):
    img_points = []
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = np.array(obj_pts, dtype=np.float32) * board_cellsize

    obj_points_all = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, board_pattern)
        if found:
            img_points.append(corners)
            obj_points_all.append(obj_points)
    
    assert len(img_points) > 0, 'No complete chessboard detected!'

    ret, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(obj_points_all, img_points, gray.shape[::-1], None, None)
    
    return ret, K, dist_coeff

if __name__ == "__main__":
    video_path = "chessboard2.avi"
    board_size = (8, 6)  # Number of internal corners (change if needed)
    cell_size = 0.03  # Real world cell size in meters (change if needed)

    images = select_img_from_video(video_path, board_size, select_all=True)
    rms, K, dist = calib_camera_from_chessboard(images, board_size, cell_size)

    print("RMS error =", rms)
    print("Camera matrix (K):\n", K)
    print("Distortion coefficients:\n", dist.ravel())