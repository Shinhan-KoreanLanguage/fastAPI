"""STT 텍스트 비교 및 통합 발음 정확도 산출 서비스.

발음 정확도 = STT 일치도(레벤슈타인 거리 기반) 와 MediaPipe 입모양 정확도를
가중 합산해 산출한다 (기본 가중치 STT 0.7 + 입모양 0.3, 요구사항 FUR-007 기준).
"""


def _levenshtein_distance(a: str, b: str) -> int:
    """두 문자열 사이의 최소 편집 거리(삽입·삭제·치환 횟수)를 계산한다."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def compute_stt_accuracy(target_text: str, recognized_text: str) -> float:
    """제시 학습 문장과 STT 인식 결과 사이의 레벤슈타인 거리 기반 발음 일치도(0~100)를 산출한다."""
    target = target_text.strip()
    recognized = recognized_text.strip()
    if not target:
        raise ValueError("비교 대상 학습 문장이 비어 있습니다.")

    distance = _levenshtein_distance(target, recognized)
    max_len = max(len(target), len(recognized), 1)
    similarity = (1 - distance / max_len) * 100
    return float(max(0.0, similarity))


def compute_final_accuracy(
    stt_accuracy: float,
    mouth_accuracy: float | None,
    stt_weight: float,
    mediapipe_weight: float,
) -> float:
    """STT 정확도와 입모양 정확도를 가중 합산해 최종 발음 정확도를 산출한다.

    입모양 분석 결과가 없는 콘텐츠(음성 전용 학습 등)는 음성 분석 결과만으로
    정확도를 산출한다 (FUR-007 요구사항 4번 항목).
    """
    if mouth_accuracy is None:
        return stt_accuracy
    return stt_accuracy * stt_weight + mouth_accuracy * mediapipe_weight


def is_correct(final_accuracy: float, pass_threshold: float) -> bool:
    """통합 발음 정확도가 합격선 이상이면 '맞은 것'으로 판정한다."""
    return final_accuracy >= pass_threshold
