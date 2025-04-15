#Camera Pose Estimation & AR Visualization

 1. Camera Calibration

- Input Video: `chessboard2.avi`
- Chessboard Pattern**: 7 x 5 (inner corners)
- Cell Size**: 30mm

Camera Intrinsic Parameters (K)
[[1959.4927, 0.0, 556.720408],  
 [0.0, 1842.37399, 437.56401],  
 [0.0, 0.0, 1.0]]
 Distortion Coefficients
[0.2785417, 0.38471694, -0.02475514, -0.13305862, -0.89482082]

 Reprojection Error (RMSE)

0.4484

---

2.Camera Pose Estimation

- 체스보드의 3D 좌표와 이미지에서 검출한 2D 코너 정보를 이용해 `cv.solvePnP()`를 사용하여 포즈 추정.
- 추정된 회전 벡터(`rvec`)와 이동 벡터(`tvec`)로부터 카메라 위치 계산 및 표시.
- 영상 내에 실시간으로 XYZ 좌표 텍스트 출력.

---
3.Augmented Reality (AR) Visualization

- OpenCV의 `cv.projectPoints()`를 활용해 3D 박스를 체스보드 위에 시각적으로 표시.
- 박스는 체스보드 위의 (4,2) ~ (5,4) 영역에 생성되며, 아래쪽과 위쪽 면을 각각 빨간색과 파란색으로 그림.
- AR 박스의 옆면은 초록색 선으로 연결하여 3D 효과 부여.

---

4 결과 데모

 AR 박스가 그려진 프레임 (스크린샷)
![AR Box Example](ar_box_result.png)
<img width="1440" alt="ar_box_result" src="https://github.com/user-attachments/assets/14814714-4442-4e6f-908b-b951a17dd294" />

 결과 영상 (시연 영상)
[▶️ Click to watch demo video](ar_box_demo.mp4)  
---


https://github.com/user-attachments/assets/f6e8d89f-d9af-4368-8eb9-6682915bafb5


5.실행 방법

```bash
python pose_estimation_with_ar.py
