"""Whisper 기반 한국어 STT(음성 -> 텍스트) 서비스.

사용 모델: faster-whisper의 small (CTranslate2 변환판, int8 양자화)
GPU 없는 CPU 전용 환경 기준, 순정 transformers 구현보다 추론 속도가 체감
3~5배 빠르고 메모리도 덜 쓴다 (실측: 1회 추론 시 transformers fp32 약 1.4GB
vs faster-whisper int8 약 0.6GB). 정확도가 더 필요하면 MODEL_NAME을
"medium"/"large-v3"로 바꾸면 된다. 단, 클수록 느려진다.
점수 계산(발음 정확도 등)은 MediaPipe 입모양 분석과 결합할 때 별도로 구현한다.
"""
import os
import subprocess
import tempfile
from functools import lru_cache

import imageio_ffmpeg
import numpy as np
from faster_whisper import WhisperModel

# 오디오 디코딩(wav/mp3/m4a 등)에 쓸 ffmpeg 실행 파일 경로.
# 시스템에 ffmpeg를 따로 설치하지 않아도 pip으로 받은 정적 바이너리를 사용한다.
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

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


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    """wav/mp3/m4a 등 다양한 포맷의 오디오 바이트를 16kHz 모노 파형으로 변환한다.

    m4a/mp4 계열 컨테이너는 메타데이터(moov atom)가 파일 끝에 저장되는 경우가
    많아, 탐색(seek)이 불가능한 표준입력 파이프로 넘기면 ffmpeg가 오디오를
    전혀 못 읽고 빈 결과를 내는 경우가 있다. 이를 피하기 위해 업로드된
    바이트를 임시 파일에 써서 경로로 넘긴다(파일은 탐색이 가능하므로 안전).
    출력은 굳이 탐색이 필요 없으므로 표준출력 파이프로 받는다.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    command = [
        FFMPEG_EXE,
        "-hide_banner",
        "-loglevel", "error",
        "-i", tmp_path,       # 임시 파일 경로로 입력 (탐색 가능)
        "-f", "s16le",        # 출력 포맷: 헤더 없는 16bit PCM
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",           # 모노로 다운믹스
        "pipe:1",             # 표준출력으로 디코딩 결과를 내보냄
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        # ffmpeg가 디코딩에 실패한 경우(깨진 파일, 지원하지 않는 코덱 등) stderr를 그대로 노출
        raise ValueError(e.stderr.decode(errors="ignore")) from e
    finally:
        os.remove(tmp_path)

    if not result.stdout:
        raise ValueError("오디오에서 소리를 추출하지 못했습니다. 오디오 트랙이 없거나 손상된 파일일 수 있습니다.")

    # 16bit 정수 PCM 바이트를 모델이 기대하는 [-1, 1] 범위의 float32로 정규화
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def transcribe(audio_bytes: bytes) -> str:
    """오디오 바이트(wav/mp3/m4a 등)를 텍스트로 변환한다."""
    # 캐싱된 모델을 가져옴 (최초 호출 시에만 실제로 다운로드/로드됨)
    model = _load_model()

    waveform = _decode_audio(audio_bytes)

    # language/task를 한국어/전사로 고정해서 언어 자동판별 오류를 방지.
    # segments는 문장 구간별 제너레이터라 순회하며 이어붙여야 전체 텍스트가 나온다.
    segments, _info = model.transcribe(waveform, language="ko", task="transcribe")
    return "".join(segment.text for segment in segments).strip()
