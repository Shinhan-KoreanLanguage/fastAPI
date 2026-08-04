"""오디오 피치(억양) 기반 음성 시각화 서비스 (VIZ-001, FUR-008/010/011).

원어민의 억양 곡선(시간별 피치)과 학습자의 억양 곡선을 DTW로 비교해,
프론트가 두 파형을 겹쳐 그릴 수 있는 데이터, 유사도 점수, 오차 구간
하이라이트, 음절별 점수를 만든다.

주의: STT 정확도·MediaPipe 입모양 정확도와 달리 "발음이 맞았는지"를
판정하는 지표가 아니라 억양 참고용 시각화 지표이므로,
pronunciation_service.compute_final_accuracy(게임 합/불합격 판정)에는
반영하지 않는다. 개인·사투리 차이가 큰 영역을 오답 판정에 섞으면
정당한 발음도 억양이 다르다는 이유로 틀렸다고 판정될 수 있기 때문이다.
"""
import librosa
import numpy as np

from app.services.audio_utils import decode_audio_to_waveform

SAMPLE_RATE = 16000
# librosa.pyin 프레임 홉 길이(샘플). 16kHz 기준 약 32ms 간격으로 피치를 추출한다.
HOP_LENGTH = 512
# 사람 목소리가 나오는 대역(약 65Hz~2093Hz)만 탐색해 옥타브 오검출을 줄인다.
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C7")


def extract_pitch_sequence(audio_bytes: bytes) -> list[dict]:
    """오디오(또는 영상) 바이트에서 유성 구간의 피치를 시간·세미톤 쌍의 시계열로 추출한다.

    반환 형식: [{"time": 0.032, "pitch": 1.05}, ...] (time은 오디오 시작 기준 초 단위)

    무음/무성음 구간(피치가 검출되지 않는 프레임)은 결과에서 제외하지만,
    유성 구간의 time 값은 원래 프레임 인덱스(hop_length 기준)를 그대로
    사용해 실제 시각을 보존한다 - 프레임을 건너뛰고 압축만 하면 나중에
    "몇 초 구간이 틀렸다"는 하이라이트(FUR-010)나 음절별 시간 매핑
    (FUR-011)을 만들 때 시간축이 틀어지기 때문이다.

    화자마다 다른 목소리 톤(성인 남녀 등) 차이를 지우고 억양의 굴곡만
    남기기 위해, 발화 전체의 중앙값 피치를 기준으로 세미톤(반음) 단위로
    정규화한다.
    """
    waveform = decode_audio_to_waveform(audio_bytes, SAMPLE_RATE)

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        waveform, fmin=FMIN, fmax=FMAX, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    voiced_indices = np.nonzero(voiced_flag)[0]
    if voiced_indices.size == 0:
        return []

    voiced_f0 = f0[voiced_indices]
    median_f0 = float(np.median(voiced_f0))
    semitones = 12.0 * np.log2(voiced_f0 / median_f0)
    times = voiced_indices * HOP_LENGTH / SAMPLE_RATE

    return [{"time": float(t), "pitch": float(p)} for t, p in zip(times, semitones)]


def _dtw_align(values_a: list[float], values_b: list[float]) -> tuple[float, list[tuple[int, int]]]:
    """두 세미톤 시퀀스 간 DTW 거리와 정렬 경로(인덱스 쌍 목록)를 함께 계산한다.

    거리만 필요했던 이전 버전(_dtw_distance)과 달리, 경로 자체가 있어야
    "사용자 시퀀스의 i번째 프레임이 원어민 시퀀스의 몇 번째 프레임과
    대응하는지"를 알 수 있고, 그래야 오차 구간 하이라이트(FUR-010)와
    음절별 구간 분할(FUR-011)을 만들 수 있다.
    """
    n, m = len(values_a), len(values_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance = abs(values_a[i - 1] - values_b[j - 1])
            cost[i, j] = distance + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        candidates = []
        if i > 0 and j > 0:
            candidates.append((cost[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((cost[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((cost[i, j - 1], i, j - 1))
        _, i, j = min(candidates)
    path.reverse()

    avg_distance = cost[n, m] / (n + m)  # 경로 길이로 정규화해 시퀀스 길이 차이의 영향을 줄임
    return float(avg_distance), path


def compute_pitch_similarity(user_curve: list[dict], reference_curve: list[dict], decay: float = 0.3) -> float:
    """DTW 거리를 0~100점의 억양 유사도 점수로 변환한다.

    decay: 평균 오차(세미톤)가 점수에 얼마나 민감하게 반영될지 조절하는
    지수 감쇠 상수 (settings.pitch_similarity_decay로 조정).
    """
    user_values = [p["pitch"] for p in user_curve]
    reference_values = [p["pitch"] for p in reference_curve]
    if not user_values or not reference_values:
        raise ValueError("억양 비교를 위한 피치 시퀀스가 비어 있습니다.")

    avg_distance, _path = _dtw_align(user_values, reference_values)
    similarity = 100.0 * np.exp(-decay * avg_distance)
    return float(np.clip(similarity, 0.0, 100.0))


def compare_pitch_curves(
    user_curve: list[dict],
    reference_curve: list[dict],
    decay: float,
    error_threshold: float,
    length_mismatch_ratio: float,
) -> dict:
    """FUR-010(피치 유사도 시각화): 유사도 점수 + 오차 구간 하이라이트를 함께 산출한다.

    두 시퀀스의 길이 차이가 length_mismatch_ratio를 넘으면(예: 발화 하나가
    통째로 누락됨) 정상적인 시간축 정렬이 불가능하다고 보고, 비교를 생략하고
    length_mismatch=True만 반환한다 (요구사항 5번: "시각화를 중단하고 안내
    메시지를 표시해야 한다" - 프론트는 이 플래그를 보고 그래프 대신 안내
    문구를 띄우면 된다).

    하이라이트 구간은 사용자 녹음 기준 시간축(user_curve의 time)으로 반환한다.
    DTW 경로에서 사용자 프레임 i에 매칭된 원어민 프레임들과의 평균 오차가
    error_threshold(세미톤)를 넘는 연속 구간을 하나의 하이라이트로 묶는다.
    """
    user_values = [p["pitch"] for p in user_curve]
    reference_values = [p["pitch"] for p in reference_curve]
    if not user_values or not reference_values:
        raise ValueError("억양 비교를 위한 피치 시퀀스가 비어 있습니다.")

    longer, shorter = max(len(user_values), len(reference_values)), min(len(user_values), len(reference_values))
    if longer / shorter > length_mismatch_ratio:
        return {"similarity": None, "length_mismatch": True, "highlight_segments": None}

    avg_distance, path = _dtw_align(user_values, reference_values)
    similarity = float(np.clip(100.0 * np.exp(-decay * avg_distance), 0.0, 100.0))

    errors_by_user_index: dict[int, list[float]] = {}
    for i, j in path:
        errors_by_user_index.setdefault(i, []).append(abs(user_values[i] - reference_values[j]))

    highlight_segments = []
    current_start = None
    for i in range(len(user_values)):
        avg_error = sum(errors_by_user_index.get(i, [0.0])) / len(errors_by_user_index.get(i, [1]))
        if avg_error > error_threshold:
            if current_start is None:
                current_start = user_curve[i]["time"]
        elif current_start is not None:
            highlight_segments.append({"start_time": current_start, "end_time": user_curve[i - 1]["time"]})
            current_start = None
    if current_start is not None:
        highlight_segments.append({"start_time": current_start, "end_time": user_curve[-1]["time"]})

    return {"similarity": similarity, "length_mismatch": False, "highlight_segments": highlight_segments}


def score_syllables(
    user_curve: list[dict], reference_curve: list[dict], target_text: str, decay: float, pass_threshold: float
) -> list[dict]:
    """FUR-011(구간별 정밀 분석): target_text의 음절 하나하나에 시간 구간과 점수를 매핑한다.

    한글은 완성형 글자 하나(공백 제외)가 곧 음절 하나이므로, target_text에서
    공백을 제거한 각 글자를 음절로 취급한다. 음절 단위 강제 정렬(forced
    alignment) 없이 DTW 경로를 음절 개수만큼 균등 분할하는 근사 방식을
    쓴다 - 실제 발화에서 음절 길이가 완전히 균등하지는 않지만, 별도의
    음성-텍스트 정렬 모델 없이는 이 이상의 정밀도를 낼 수 없어 이 프로젝트
    범위에서는 합리적인 근사로 판단했다.
    """
    user_values = [p["pitch"] for p in user_curve]
    reference_values = [p["pitch"] for p in reference_curve]
    if not user_values or not reference_values:
        raise ValueError("억양 비교를 위한 피치 시퀀스가 비어 있습니다.")

    syllables = [ch for ch in target_text if not ch.isspace()]
    if not syllables:
        return []

    _avg_distance, path = _dtw_align(user_values, reference_values)

    boundaries = np.linspace(0, len(path), len(syllables) + 1, dtype=int)
    results = []
    for idx, syllable in enumerate(syllables):
        chunk = path[boundaries[idx]:boundaries[idx + 1]] or [path[-1]]
        avg_error = sum(abs(user_values[i] - reference_values[j]) for i, j in chunk) / len(chunk)
        score = float(np.clip(100.0 * np.exp(-decay * avg_error), 0.0, 100.0))
        user_indices = [i for i, _ in chunk]
        results.append({
            "syllable": syllable,
            "score": round(score, 2),
            "is_weak": score < pass_threshold,
            "start_time": user_curve[min(user_indices)]["time"],
            "end_time": user_curve[max(user_indices)]["time"],
        })
    return results
