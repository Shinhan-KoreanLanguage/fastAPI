# -*- coding: utf-8 -*-
"""원어민 발음 영상 촬영 도구 (배경 블러 + 음성 동시 녹음).

웹캠 화면의 배경만 흐리게 처리하면서 녹화한다. 결과물은 두 개다.

  <이름>.mp4  영상(소리 포함) - 학습 화면에서 재생하고, 서버가 입모양 랜드마크를 뽑는다
  <이름>.wav  음성만          - 서버가 억양(피치) 곡선을 뽑는다

앱이 영상과 음성을 별도 파일로 받도록 되어 있어(register_reference 참고)
한 번 촬영으로 두 파일을 함께 만들어 둔다.

사용법:
    python scripts/record_reference.py --text 사과
    python scripts/record_reference.py --text 사과 --blur 45 --camera 0

조작:
    스페이스  녹화 시작 / 정지 (정지하면 파일로 저장)
    Q        종료

배경 분리는 MediaPipe selfie segmentation을 쓴다. 서버가 입모양 분석에 쓰는 것과
같은 라이브러리라, 촬영 중 얼굴 인식 여부를 화면에서 바로 확인할 수 있다.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# 촬영 기본값. 서버가 프레임마다 MediaPipe를 돌리므로 과한 해상도는 분석만 느려진다.
WIDTH, HEIGHT, FPS = 1280, 720, 30

# 얼굴 인식 확인용 - 서버의 mediapipe_service와 같은 인덱스(입꼬리·입술 중앙)
MOUTH_INDICES = (61, 291, 13, 14)


def build_arg_parser():
    p = argparse.ArgumentParser(description="원어민 발음 영상 촬영 (배경 블러)")
    p.add_argument("--text", required=True, help="학습 단어/문장. 파일 이름과 화면 안내에 쓰인다")
    p.add_argument("--outdir", default="recordings", help="저장 폴더 (기본: recordings)")
    p.add_argument("--camera", type=int, default=0, help="웹캠 인덱스 (기본 0, 화면이 안 나오면 1로)")
    p.add_argument("--mic", default="마이크(USB 2.0 Camera)", help="마이크 장치 이름")
    p.add_argument("--blur", type=int, default=35, help="배경 블러 강도, 홀수 (기본 35)")
    return p


def start_audio_capture(mic_name: str, wav_path: Path) -> subprocess.Popen:
    """마이크 녹음을 ffmpeg 자식 프로세스로 시작한다.

    OpenCV의 VideoWriter는 영상만 쓸 수 있어 소리가 빠진다. 그래서 음성은
    ffmpeg가 따로 받아두고, 녹화가 끝난 뒤 영상과 합친다.
    """
    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "dshow", "-i", "audio=%s" % mic_name,
        "-ar", "44100", "-ac", "1",
        str(wav_path),
    ]
    # stdin을 열어두는 이유: 종료할 때 'q'를 보내 ffmpeg가 헤더를 정상적으로 마무리하게 한다.
    return subprocess.Popen(command, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stop_audio_capture(proc: subprocess.Popen) -> bool:
    try:
        proc.communicate(input=b"q", timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    return proc.returncode == 0


def mux(video_path: Path, wav_path: Path, out_path: Path) -> bool:
    """무음 영상과 음성을 하나의 mp4로 합친다 (재인코딩 없이 스트림 복사)."""
    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-i", str(wav_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_path),
    ]
    return subprocess.run(command, capture_output=True).returncode == 0


def blur_background(frame, mask, kernel: int):
    """사람 영역은 그대로 두고 배경만 흐리게 만든다."""
    blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
    # 마스크 경계를 부드럽게 해서 윤곽선이 딱딱하게 잘리는 것을 막는다
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)[..., None]
    return (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)


def has_face(face_mesh, frame_rgb) -> bool:
    result = face_mesh.process(frame_rgb)
    if not result.multi_face_landmarks:
        return False
    landmarks = result.multi_face_landmarks[0].landmark
    return len(landmarks) > max(MOUTH_INDICES)


def draw_hud(frame, text, recording, elapsed, face_ok):
    h = frame.shape[0]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - 34), (frame.shape[1], h), (0, 0, 0), -1)

    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if face_ok:
        status, color = "FACE OK", (0, 220, 0)
    else:
        status, color = "NO FACE - move into frame", (0, 120, 255)
    cv2.putText(frame, status, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if recording:
        cv2.circle(frame, (frame.shape[1] - 130, 20), 9, (0, 0, 255), -1)
        cv2.putText(frame, "REC %4.1fs" % elapsed, (frame.shape[1] - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "SPACE=rec  Q=quit", (frame.shape[1] - 250, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    args = build_arg_parser().parse_args()
    kernel = args.blur if args.blur % 2 == 1 else args.blur + 1

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, FPS)
    if not capture.isOpened():
        sys.exit("웹캠을 열지 못했습니다. --camera 값을 1로 바꿔 다시 시도해 보세요.")

    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    writer = None
    audio_proc = None
    tmp_video = tmp_wav = None
    started_at = 0.0
    saved = []

    print("준비됐습니다. 미리보기 창에서 스페이스=녹화, Q=종료")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break

            frame = cv2.flip(frame, 1)  # 거울처럼 보이게 (촬영자가 움직임을 맞추기 쉽다)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            seg = segmenter.process(frame_rgb)
            mask = seg.segmentation_mask > 0.5
            composited = blur_background(frame, mask, kernel)

            if writer is not None:
                writer.write(composited)

            preview = composited.copy()
            draw_hud(preview, args.text, writer is not None,
                     time.time() - started_at if writer else 0.0,
                     has_face(face_mesh, frame_rgb))
            cv2.imshow("record_reference  (SPACE=rec, Q=quit)", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if writer is None:
                    stamp = datetime.now().strftime("%H%M%S")
                    base = "%s_%s" % (args.text.replace(" ", "_"), stamp)
                    tmp_video = outdir / ("%s_silent.mp4" % base)
                    tmp_wav = outdir / ("%s.wav" % base)
                    writer = cv2.VideoWriter(str(tmp_video),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             FPS, (frame.shape[1], frame.shape[0]))
                    audio_proc = start_audio_capture(args.mic, tmp_wav)
                    started_at = time.time()
                    print("  녹화 시작")
                else:
                    writer.release()
                    writer = None
                    ok_audio = stop_audio_capture(audio_proc)
                    audio_proc = None

                    final = outdir / tmp_video.name.replace("_silent", "")
                    if ok_audio and tmp_wav.exists() and mux(tmp_video, tmp_wav, final):
                        tmp_video.unlink(missing_ok=True)
                        print("  저장: %s" % final)
                        print("        %s" % tmp_wav)
                        saved.append(final)
                    else:
                        # 소리가 안 잡혀도 영상은 살린다 - 입모양 분석에는 소리가 필요 없다
                        tmp_video.rename(final)
                        print("  저장(무음): %s" % final)
                        print("  [경고] 마이크 녹음에 실패했습니다. --mic 값을 확인하세요.")
                        saved.append(final)
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
        if audio_proc is not None:
            stop_audio_capture(audio_proc)
        capture.release()
        cv2.destroyAllWindows()
        segmenter.close()
        face_mesh.close()

    print("\n촬영 완료. 저장 위치: %s" % outdir)
    for path in saved:
        print("  - %s" % path.name)


if __name__ == "__main__":
    main()
