# 404-ai (간단 안내)

404-ai는 Flask 기반의 이미지 이상 탐지(Anomaly Detection) 서버입니다. YOLO / FastSAM을 이용한 객체 감지와 PatchCore 기반의 이상점수 산출을 결합해 공정 상 결함을 검출하고, 결과를 HTTP 응답 및 MQTT로 게시합니다.

핵심 기능
- REST API로 이미지 업로드/검출 제공
- YOLO 또는 FastSAM 감지 백엔드 선택 가능
- PatchCore 기반 애노말리(이상) 검출 백엔드
- MQTT로 결과 발행(옵션)
- 환경변수로 구성 가능한 튜닝 파라미터

프로젝트 구조(주요 파일)

```
404-ai/
├─ app.py                # 메인 Flask 앱, 엔드포인트 및 MQTT 연동
├─ pipeline.py           # 감지 + 애노말리 파이프라인 로직
├─ detection/            # detector 래퍼 (yolo, sam)
├─ anomaly/              # anomaly backend (patchcore, PDN 등)
├─ models/               # 모델 및 체크포인트 저장
├─ img/                  # 예제 이미지
├─ logs/                 # 실행 로그
├─ debug/                # 임시 업로드 세션
├─ requirements.txt
└─ README.md
```

빠른 시작
1) 의존성 설치
```bash
pip install -r requirements.txt
```

2) 환경파일 복사 및 편집
```bash
copy .env.example .env
# .env에서 MQTT, 모델 경로, DEVICE 등 설정
```

3) 서버 실행
```bash
python app.py
```

기본 동작
- 서버: 기본 호스트 `0.0.0.0`, 포트 `5000` (환경변수 `HOST`, `PORT`로 변경 가능)
- 감지 파이프라인 구성은 `.env`의 `DETECTION_BACKEND`, `ANOMALY_BACKEND` 등으로 제어

주요 API
- `GET /` : 간단 환영 메시지
- `GET /health` : 앱 상태(업타임, 디스크 여유, MQTT 연결, detector/anomaly 준비 상태, 종속성 설치 여부)
- `POST /detect` 또는 MQTT 인입 콜백을 통해 이미지 전송 → 감지/애노말리 실행 → 결과 반환/발행

MQTT
- 활성화: `.env`에서 `MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC` 설정
- 결과 발행은 `MQTT_SEND_ACK` 등의 옵션으로 제어

환경 변수(핵심 — 권장 설정 예시)

- `DETECTION_BACKEND` : 감지 백엔드 선택. 값: `yolo` (기본) 또는 `sam`
	- 예: `DETECTION_BACKEND=yolo`
- `DETECTION_CONF` : 디폴트 모델 신뢰도(모델 레벨 필터). 예: `0.25`
	- 예: `DETECTION_CONF=0.25`
- `DETECTION_CONF_MAP` : 클래스별 최소 신뢰도 매핑. 형식 `class_id:threshold`를 쉼표로 구분.
	- 예: `DETECTION_CONF_MAP=1:0.4,2:0.6,4:0.4,5:0.3`
	- 설명: 파이프라인은 이 맵을 사용해 감지 결과를 후처리합니다. 필요 시 `conf_override`로 모델에 낮은 임계값을 요청합니다.
- `DETECTION_NMS_IOU` : NMS(중복 박스 제거) IoU 임계값(0~1). 0이면 비활성화.
	- 예: `DETECTION_NMS_IOU=0.3`
- `SAM_MODEL_PATH` : FastSAM 모델(.pt) 경로. 기본값: `models/sam/FastSAM-s.pt`
	- 예: `SAM_MODEL_PATH=models/sam/FastSAM-s.pt`
- `ANOMALY_BACKEND` : 애노말리 백엔드 선택. 값: `patchcore` (기본) 또는 `efficientad`
	- 예: `ANOMALY_BACKEND=patchcore`
- `ANOMALY_THRESHOLD` : 애노말리(이상) 스코어 임계값(백엔드별 해석 상이).
	- 예: `ANOMALY_THRESHOLD=33.08`
- `TOY_CAR_MIN_AREA_RATIO` : toy_car(클래스 3/6)으로 인정하기 위한 최소 박스 면적 비율(0~1).
	- 예: `TOY_CAR_MIN_AREA_RATIO=0.02` (이미지 면적의 2% 이상인 박스만 toy_car로 간주)
- `SHOW_IGNORED_CLASSES` : 임시 시각화용으로 기본적으로 숨긴 클래스(예: `2`)를 표시할지 여부.
	- 예: `SHOW_IGNORED_CLASSES=false`
- `MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC` 등 MQTT 관련 설정
	- 예:
		- `MQTT_BROKER=localhost`
		- `MQTT_PORT=1883`
		- `MQTT_TOPIC=camera01/result`
- `MQTT_SEND_ACK` : 결과 ACK 발행 여부 (0/1)

참고
- `.env.example` 파일에 위 변수 일부를 미리 채워두면 편리합니다. 설정값을 변경할 때는 우선 테스트 환경에서 감도(특히 `DETECTION_CONF_MAP`과 `TOY_CAR_MIN_AREA_RATIO`)를 확인하세요.

테스트 및 디버그
- 로컬에서 `/health` 호출로 기본 준비 상태 확인
```bash
curl http://127.0.0.1:5000/health
```

권장: 실제 MQTT 브로커와 모델 파일이 준비된 환경에서 E2E(감지→애노말리→MQTT 발행) 검증을 권장합니다.

기여 및 라이선스
- 개발/배포 전 보안(Secret, HTTPS), 인증, 환경 구성 확인 권장
- 라이선스 및 기여 가이드는 프로젝트 루트에 추가하세요.

문의
- 코드 관련 문의 또는 특정 기능 추가 요청이 있으면 알려주세요.