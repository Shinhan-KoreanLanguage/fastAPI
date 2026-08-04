"""MediaPipe 기반 입모양(입술) 분석 서비스.

입모양 데이터는 두 가지 경로로 받을 수 있다.
1) 클라이언트(MediaPipe.js)가 녹화 중 이미 추출해 둔 468개 얼굴 랜드마크 좌표를
   프레임 단위 JSON 배열로 전달하는 경우 (parse_landmark_sequence)
2) 서버가 웹캠 녹화 영상 파일을 받아 mediapipe(Python) FaceMesh로 직접
   랜드마크를 추출하는 경우 (extract_lip_sequence_from_video)

두 경로 모두 동일한 정규화·DTW 비교 로직(compute_mouth_accuracy)을 공유한다.
"""
import os
import tempfile

import cv2
import mediapipe as mp
import numpy as np

_face_mesh_solution = mp.solutions.face_mesh

# 얼굴 기준점 인덱스 (468 랜드마크 모델에서 널리 쓰이는 코끝·양 눈 바깥쪽 코너·입꼬리·입술 안쪽 중앙).
NOSE_TIP_INDEX = 1
LEFT_EYE_OUTER_INDEX = 33
RIGHT_EYE_OUTER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291
UPPER_INNER_LIP_INDEX = 13
LOWER_INNER_LIP_INDEX = 14

_REQUIRED_INDICES = [
    NOSE_TIP_INDEX,
    LEFT_EYE_OUTER_INDEX,
    RIGHT_EYE_OUTER_INDEX,
    LEFT_MOUTH_CORNER_INDEX,
    RIGHT_MOUTH_CORNER_INDEX,
    UPPER_INNER_LIP_INDEX,
    LOWER_INNER_LIP_INDEX,
]
_MIN_LANDMARK_COUNT = max(_REQUIRED_INDICES) + 1


def _extract_mouth_features(landmark_xy: np.ndarray) -> np.ndarray:
    """전체 468개 랜드마크 좌표(Nx2)를 입모양을 요약하는 두 개의 무차원 비율로 변환한다.

    원본 좌표(위치)를 그대로 비교하면 화자의 얼굴·입 크기 차이가 그대로 오차에 섞여
    다른 사람이 찍은 영상끼리 비교할 때 정확도가 왜곡된다. 대신 아래 두 비율만 비교하면
    화자가 달라도 "입을 얼마나, 어떤 모양으로 움직였는지"만 남는다.
    - open_ratio: 입이 수직으로 벌어진 정도 (눈 사이 거리 대비, 촬영 거리 보정용)
    - aspect_ratio: 입의 세로/가로 비율 (입 자체 크기에 대해 스케일 불변이라 개인차 영향이 적음.
      너비 대신 높이를 분자로 둔 이유는 입을 다물면 분모(너비)는 0에 가까워지지 않지만
      분자(높이)는 0에 가까워져, width/height로 두면 나눗셈이 불안정해지기 때문)
    """
    eye_distance = np.linalg.norm(landmark_xy[RIGHT_EYE_OUTER_INDEX] - landmark_xy[LEFT_EYE_OUTER_INDEX])
    eye_distance = eye_distance if eye_distance > 1e-6 else 1e-6

    mouth_width = np.linalg.norm(landmark_xy[RIGHT_MOUTH_CORNER_INDEX] - landmark_xy[LEFT_MOUTH_CORNER_INDEX])
    mouth_width = mouth_width if mouth_width > 1e-6 else 1e-6
    mouth_height = np.linalg.norm(landmark_xy[LOWER_INNER_LIP_INDEX] - landmark_xy[UPPER_INNER_LIP_INDEX])

    open_ratio = mouth_height / eye_distance
    aspect_ratio = mouth_height / mouth_width
    return np.array([open_ratio, aspect_ratio], dtype=np.float64)


def parse_landmark_sequence(frames: list[list[list[float]]]) -> list[np.ndarray]:
    """클라이언트가 전송한 프레임별 468개 랜드마크 좌표(JSON)를 입모양 특징(open_ratio, aspect_ratio) 시퀀스로 변환한다.

    각 프레임은 mediapipe Face Mesh와 동일한 인덱스 순서의 [x, y] (또는 [x, y, z]) 468개 좌표여야 한다.
    """
    sequence: list[np.ndarray] = []
    for frame in frames:
        if len(frame) < _MIN_LANDMARK_COUNT:
            # 기준점(코·눈·입꼬리·입술) 인덱스를 구성하기에 좌표 수가 부족한 프레임은 건너뜀
            continue
        landmark_xy = np.array([[point[0], point[1]] for point in frame], dtype=np.float64)
        sequence.append(_extract_mouth_features(landmark_xy))
    return sequence


def extract_raw_landmarks_from_video(video_bytes: bytes) -> list[list[list[float]]]:
    """웹캠 녹화 영상 파일에서 mediapipe(Python) FaceMesh로 프레임별 원본(정규화 이전) 468개
    랜드마크 좌표를 추출한다.

    parse_landmark_sequence 가 기대하는 것과 동일한 원본 좌표 형식으로 반환하므로,
    반환값을 그대로 JSON 저장해두면 이후 reference_landmarks/landmarks 값으로 재사용할 수 있다.
    (예: 원어민 발음 영상을 1회성으로 처리해 표준 시퀀스 DB를 구축할 때 사용)
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames: list[list[list[float]]] = []
    try:
        capture = cv2.VideoCapture(tmp_path)
        with _face_mesh_solution.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(frame_rgb)
                if not result.multi_face_landmarks:
                    # 얼굴/입모양 인식 실패 프레임은 건너뜀 (FUR-005 요구사항 5번 항목)
                    continue
                landmarks = result.multi_face_landmarks[0].landmark
                frames.append([[point.x, point.y] for point in landmarks])
        capture.release()
    finally:
        os.remove(tmp_path)

    return frames


def extract_lip_sequence_from_video(video_bytes: bytes) -> list[np.ndarray]:
    """웹캠 녹화 영상 파일에서 입모양 특징(open_ratio, aspect_ratio) 시퀀스를 추출한다.

    원본 랜드마크 추출(extract_raw_landmarks_from_video) 후 parse_landmark_sequence 와
    동일한 특징 추출 로직을 거치므로, 사용자 랜드마크 JSON을 처리하는 경로와 결과가 일관된다.
    """
    raw_frames = extract_raw_landmarks_from_video(video_bytes)
    return parse_landmark_sequence(raw_frames)


def _dtw_distance(seq_a: list[np.ndarray], seq_b: list[np.ndarray]) -> float:
    """두 입모양 특징(open_ratio, aspect_ratio) 시퀀스 간 동적 시간 와핑(DTW) 거리를 계산한다.

    사용자와 원어민의 발화 속도가 서로 달라 프레임 수가 달라도(시간축 어긋남),
    입모양 형태 자체의 유사도를 비교할 수 있도록 DTW로 시간축을 정렬한다.
    """
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            cost[i, j] = distance + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / (n + m)  # 경로 길이로 정규화해 시퀀스 길이 차이의 영향을 줄임


def compute_mouth_accuracy(
    user_sequence: list[np.ndarray], reference_sequence: list[np.ndarray], decay: float = 3.0
) -> float:
    """DTW 거리를 0~100점의 입모양 정확도 점수로 변환한다.

    decay: 평균 오차가 점수에 얼마나 민감하게 반영될지 조절하는 지수 감쇠 상수.
    클수록 같은 오차에도 점수가 더 가파르게 떨어진다 (기본 3.0). 채점이 너무
    박하면 낮추고, 너무 후하면 올린다 (settings.mouth_accuracy_decay 로 조정).
    """
    if not user_sequence or not reference_sequence:
        raise ValueError("입모양 비교를 위한 좌표 시퀀스가 비어 있습니다.")

    avg_distance = _dtw_distance(user_sequence, reference_sequence)
    # 무차원 비율 특징(open_ratio, aspect_ratio) 기준 평균 오차가 클수록 점수가 지수적으로 감소하도록 매핑
    accuracy = 100.0 * np.exp(-decay * avg_distance)
    return float(np.clip(accuracy, 0.0, 100.0))
