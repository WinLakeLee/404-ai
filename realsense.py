import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 파이프라인 설정 (카메라 연결 준비)
pipeline = rs.pipeline()
config = rs.config()

# 2. 스트림 설정 (해상도 640x480, 30프레임)
# Depth: 거리 정보 (Z16 포맷)
# Color: RGB 정보 (BGR8 포맷 - OpenCV는 RGB 대신 BGR을 씁니다)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 3. 카메라 시작
print("카메라를 시작합니다... (종료하려면 화면을 클릭하고 'q'를 누르세요)")
pipeline.start(config)

try:
    while True:
        # 4. 프레임 기다리기 (Color와 Depth가 모두 들어올 때까지 대기)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 5. 데이터를 numpy 배열로 변환
        # Depth는 0~65535(mm) 값을 가지므로 그냥 보면 검은색으로만 보입니다.
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 6. Depth 이미지를 컬러맵으로 변환 (시각화용)
        # alpha=0.03은 거리에 따라 색상 변화를 잘 보여주기 위한 스케일링 값입니다.
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 7. 두 이미지를 가로로 합치기 (왼쪽: RGB, 오른쪽: Depth)
        images = np.hstack((color_image, depth_colormap))

        # 8. 화면에 띄우기
        cv2.imshow('RealSense', images)

        # 'q' 키나 'ESC'를 누르면 종료
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

finally:
    # 9. 종료 시 카메라 정지
    pipeline.stop()
    cv2.destroyAllWindows()