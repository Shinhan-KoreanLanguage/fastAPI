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

# MediaPipe Face Mesh는 얼굴당 468개 랜드마크를 반환한다.
# 입술 주변 랜드마크 인덱스는 라이브러리가 제공하는 연결 그래프(FACEMESH_LIPS)에서 유도한다.
LIP_LANDMARK_INDICES = sorted({idx for pair in _face_mesh_solution.FACEMESH_LIPS for idx in pair})

# 얼굴 기준점 인덱스 (468 랜드마크 모델에서 널리 쓰이는 코끝·양 눈 바깥쪽 코너).
# 카메라와 사용자 사이 거리·각도 차이를 보정하는 정규화 기준으로 사용한다.
NOSE_TIP_INDEX = 1
LEFT_EYE_OUTER_INDEX = 33
RIGHT_EYE_OUTER_INDEX = 263

_REQUIRED_INDICES = LIP_LANDMARK_INDICES + [NOSE_TIP_INDEX, LEFT_EYE_OUTER_INDEX, RIGHT_EYE_OUTER_INDEX]
_MIN_LANDMARK_COUNT = max(_REQUIRED_INDICES) + 1


def _normalize_points(points: np.ndarray, nose: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> np.ndarray:
    """코를 원점으로, 양 눈 사이 거리를 척도로 좌표를 정규화한다.

    사용자와 카메라 사이 거리·얼굴 크기가 달라도 입모양 형태 자체만 일관되게 비교하기 위함.
    """
    eye_distance = np.linalg.norm(right_eye - left_eye)
    scale = eye_distance if eye_distance > 1e-6 else 1e-6
    return (points - nose) / scale


def _extract_lip_frame(landmark_xy: np.ndarray) -> np.ndarray:
    """전체 468개 랜드마크 좌표(Nx2)에서 입술 좌표만 정규화해 1차원 벡터로 반환한다."""
    nose = landmark_xy[NOSE_TIP_INDEX]
    left_eye = landmark_xy[LEFT_EYE_OUTER_INDEX]
    right_eye = landmark_xy[RIGHT_EYE_OUTER_INDEX]
    lip_points = landmark_xy[LIP_LANDMARK_INDICES]
    return _normalize_points(lip_points, nose, left_eye, right_eye).flatten()


def parse_landmark_sequence(frames: list[list[list[float]]]) -> list[np.ndarray]:
    """클라이언트가 전송한 프레임별 468개 랜드마크 좌표(JSON)를 정규화된 입술 시퀀스로 변환한다.

    각 프레임은 mediapipe Face Mesh와 동일한 인덱스 순서의 [x, y] (또는 [x, y, z]) 468개 좌표여야 한다.
    """
    sequence: list[np.ndarray] = []
    for frame in frames:
        if len(frame) < _MIN_LANDMARK_COUNT:
            # 기준점(코·눈) 또는 입술 인덱스를 구성하기에 좌표 수가 부족한 프레임은 건너뜀
            continue
        landmark_xy = np.array([[point[0], point[1]] for point in frame], dtype=np.float64)
        sequence.append(_extract_lip_frame(landmark_xy))
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
    """웹캠 녹화 영상 파일에서 정규화된 입술 좌표 시퀀스를 추출한다.

    원본 랜드마크 추출(extract_raw_landmarks_from_video) 후 parse_landmark_sequence 와
    동일한 정규화 로직을 거치므로, 사용자 랜드마크 JSON을 처리하는 경로와 결과가 일관된다.
    """
    raw_frames = extract_raw_landmarks_from_video(video_bytes)
    return parse_landmark_sequence(raw_frames)


def _dtw_distance(seq_a: list[np.ndarray], seq_b: list[np.ndarray]) -> float:
    """두 입술 좌표 시퀀스 간 동적 시간 와핑(DTW) 거리를 계산한다.

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
    # 정규화된 좌표계(척도: 눈 사이 거리)에서 평균 오차가 클수록 점수가 지수적으로 감소하도록 매핑
    accuracy = 100.0 * np.exp(-decay * avg_distance)
    return float(np.clip(accuracy, 0.0, 100.0))
