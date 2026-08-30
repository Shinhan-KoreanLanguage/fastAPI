"""Whisper 기반 한국어 STT(음성 -> 텍스트) 서비스.

사용 모델: faster-whisper (CTranslate2 변환판, int8 양자화)
GPU 없는 CPU 전용 환경 기준, 순정 transformers 구현보다 추론 속도가 체감
3~5배 빠르고 메모리도 덜 쓴다 (실측: 1회 추론 시 transformers fp32 약 1.4GB
vs faster-whisper int8 약 0.6GB).

용도에 따라 모델을 나눠 쓴다.
- DETAIL_MODEL(small): 발음 연습(/analyze). 문장 단위 피드백이라 정확도가 중요하고,
  사용자도 결과 화면을 기다리는 맥락이라 2초 남짓은 감수할 수 있다.
- GAME_MODEL(base): 게임(/score). 단어 하나가 맞았는지만 70% 임계값으로 판정하므로
  small만큼의 정확도가 필요 없다. 대신 응답이 느리면 게임 자체가 성립하지 않는다.

Whisper는 입력을 30초 창으로 패딩해 인코더를 돌리기 때문에, 2초짜리 단어도
5초짜리 문장과 추론 시간이 거의 같다. 즉 오디오를 짧게 잘라도 빨라지지 않으므로
게임 응답 속도는 모델 크기로 확보해야 한다 (실측 small 2.31s / base 0.71s / tiny 0.37s).
게임 오판정이 잦으면 GAME_MODEL을 small로 되돌리고, 더 빨라야 하면 tiny로 낮춘다.
"""
from functools import lru_cache

from faster_whisper import WhisperModel

from app.services.audio_utils import decode_audio_to_waveform

# 발음 연습(정확도 우선) / 게임(속도 우선) 각각에 쓸 faster-whisper 모델 이름
DETAIL_MODEL = "small"
GAME_MODEL = "base"
# Whisper가 학습된 샘플링 레이트(16kHz). 입력 오디오도 이 값으로 맞춰줘야 함
SAMPLE_RATE = 16000


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> WhisperModel:
    """모델을 이름별로 최초 1회만 로드해서 캐싱한다.

    매 요청마다 새로 로드하면 느리므로 lru_cache로 프로세스 내에서 재사용한다.
    두 모델을 동시에 들고 있어야 하므로 maxsize=2 (더 늘리면 메모리만 먹는다).
    device="cpu", compute_type="int8": GPU 없는 환경에서 속도/메모리 균형이 가장 좋은 조합.
    """
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes, fast: bool = False) -> str:
    """오디오 바이트(wav/mp3/m4a 등)를 텍스트로 변환한다.

    fast=True면 게임용 경량 설정으로 돌린다. 작은 모델을 쓰고, 탐색 폭을 1로
    줄이고(beam_size), 타임스탬프 토큰 생성을 생략한다 - 게임은 인식된 텍스트가
    제시 단어와 일치하는지만 보므로 구간 시각 정보가 필요 없다.
    """
    # 캐싱된 모델을 가져옴 (최초 호출 시에만 실제로 다운로드/로드됨)
    model = _load_model(GAME_MODEL if fast else DETAIL_MODEL)

    waveform = decode_audio_to_waveform(audio_bytes, SAMPLE_RATE)

    # language/task를 한국어/전사로 고정해서 언어 자동판별 오류를 방지.
    # condition_on_previous_text=False: 앞 구간 텍스트를 다음 구간 프롬프트로 넘기지
    # 않는다. 짧은 단발성 발화라 문맥 이득이 없고, 잘못 인식된 앞 구간이 뒤까지
    # 오염시키는 것을 막는다.
    # segments는 문장 구간별 제너레이터라 순회하며 이어붙여야 전체 텍스트가 나온다.
    segments, _info = model.transcribe(
        waveform,
        language="ko",
        task="transcribe",
        beam_size=1 if fast else 5,
        without_timestamps=fast,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()
