"""오디오/영상 바이트를 파형 numpy 배열로 디코딩하는 공용 유틸.

STT(stt_service)와 피치 분석(pitch_service)이 동일한 ffmpeg 디코딩 로직이
필요해서 여기로 분리했다. 영상 파일(mp4 등)을 넣어도 ffmpeg가 오디오
스트림만 자동으로 골라 디코딩하므로, 원어민 발음 영상에서 오디오만
뽑아 피치를 추출할 때도 그대로 쓸 수 있다.
"""
import os
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np

# 오디오 디코딩에 쓸 ffmpeg 실행 파일 경로.
# 시스템에 ffmpeg를 따로 설치하지 않아도 pip으로 받은 정적 바이너리를 사용한다.
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def decode_audio_to_waveform(media_bytes: bytes, sample_rate: int) -> np.ndarray:
    """wav/mp3/m4a 등 오디오는 물론 mp4 등 영상 파일에서도 오디오 트랙만 뽑아
    지정한 샘플링레이트의 모노 파형으로 변환한다.

    m4a/mp4 계열 컨테이너는 메타데이터(moov atom)가 파일 끝에 저장되는 경우가
    많아, 탐색(seek)이 불가능한 표준입력 파이프로 넘기면 ffmpeg가 오디오를
    전혀 못 읽고 빈 결과를 내는 경우가 있다. 이를 피하기 위해 업로드된
    바이트를 임시 파일에 써서 경로로 넘긴다(파일은 탐색이 가능하므로 안전).
    출력은 굳이 탐색이 필요 없으므로 표준출력 파이프로 받는다.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(media_bytes)
        tmp_path = tmp.name

    command = [
        FFMPEG_EXE,
        "-hide_banner",
        "-loglevel", "error",
        "-i", tmp_path,       # 임시 파일 경로로 입력 (탐색 가능)
        "-f", "s16le",        # 출력 포맷: 헤더 없는 16bit PCM
        "-ar", str(sample_rate),
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

    # 16bit 정수 PCM 바이트를 [-1, 1] 범위의 float32로 정규화
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0
