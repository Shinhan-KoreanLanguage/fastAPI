"""Whisper 기반 한국어 STT(음성 -> 텍스트) 서비스.

사용 모델: faster-whisper의 small (CTranslate2 변환판, int8 양자화)
GPU 없는 CPU 전용 환경 기준, 순정 transformers 구현보다 추론 속도가 체감
3~5배 빠르고 메모리도 덜 쓴다 (실측: 1회 추론 시 transformers fp32 약 1.4GB
vs faster-whisper int8 약 0.6GB). 정확도가 더 필요하면 MODEL_NAME을
"medium"/"large-v3"로 바꾸면 된다. 단, 클수록 느려진다.
점수 계산(발음 정확도 등)은 MediaPipe 입모양 분석과 결합할 때 별도로 구현한다.
"""
from functools import lru_cache

from faster_whisper import WhisperModel

from app.services.audio_utils import decode_audio_to_waveform

# faster-whisper(CTranslate2) 모델 이름
MODEL_NAME = "small"
# Whisper가 학습된 샘플링 레이트(16kHz). 입력 오디오도 이 값으로 맞춰줘야 함
SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def _load_model() -> WhisperModel:
    """모델을 최초 1회만 로드해서 캐싱한다.

    매 요청마다 새로 로드하면 느리므로 lru_cache로 프로세스 내에서 재사용한다.
    device="cpu", compute_type="int8": GPU 없는 환경에서 속도/메모리 균형이 가장 좋은 조합.
    """
    return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes) -> str:
    """오디오 바이트(wav/mp3/m4a 등)를 텍스트로 변환한다."""
    # 캐싱된 모델을 가져옴 (최초 호출 시에만 실제로 다운로드/로드됨)
    model = _load_model()

    waveform = decode_audio_to_waveform(audio_bytes, SAMPLE_RATE)

    # language/task를 한국어/전사로 고정해서 언어 자동판별 오류를 방지.
    # segments는 문장 구간별 제너레이터라 순회하며 이어붙여야 전체 텍스트가 나온다.
    segments, _info = model.transcribe(waveform, language="ko", task="transcribe")
    return "".join(segment.text for segment in segments).strip()
