import cv2 as cv
import numpy as np

# 설정: 체스보드 크기와 셀 사이즈
board_pattern = (7, 5)  # 내부 코너 수 (열, 행)
board_cellsize = 30  # mm 단위 (예: 30mm)

# 체스보드 3D 좌표 생성
obj_points = board_cellsize * np.array([
    [c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])
], dtype=np.float32)

# AR로 그릴 3D 사각 기둥 (박스) 정의
box_lower = board_cellsize * np.array([[4, 2, 0], [5, 2, 0], [5, 4, 0], [4, 4, 0]], dtype=np.float32)
box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]], dtype=np.float32)  # 위쪽은 z = -1

# 영상 파일 열기
video_file = 'chessboard2.avi'  # 본인의 영상 파일명
video = cv.VideoCapture(video_file)

#캘리브레이션 결과: 사용자 본인의 값으로 수정
K = np.array([[1959.4927, 0.0, 556.720408],
              [0.0, 1842.37399, 437.56401],
              [0.0, 0.0, 1.0]])
dist_coeff = np.array([0.2785417, 0.38471694, -0.02475514, -0.13305862, -0.89482082])

# 체스보드 코너 검출 조건
board_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    valid, img = video.read()
    if not valid:
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    complete, img_points = cv.findChessboardCorners(gray, board_pattern, None)

    if complete:
        img_points = cv.cornerSubPix(gray, img_points, (11, 11), (-1, -1), board_criteria)

        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # 3D 박스의 아래/위쪽 점을 2D로 투영
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

        # 박스 선 그리기
        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)

        # 아래쪽과 위쪽을 연결하는 선
        for b, t in zip(line_lower, line_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # 카메라 위치 추정
        R, _ = cv.Rodrigues(rvec)
        cam_position = (-R.T @ tvec).flatten()
        info = f'XYZ: [{cam_position[0]:.1f}, {cam_position[1]:.1f}, {cam_position[2]:.1f}]'
        cv.putText(img, info, (10, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

    # 결과 프레임 출력
    cv.imshow('Pose Estimation with AR Box', img)
    if cv.waitKey(1) == 27:  # ESC 키 누르면 종료
        break

video.release()
cv.destroyAllWindows()