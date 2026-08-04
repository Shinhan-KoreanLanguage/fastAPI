"""원어민 발음 영상에서 표준 입모양 랜드마크 시퀀스(JSON)를 추출하는 1회성 스크립트.

학습 콘텐츠(단어/문장) 하나당 원어민이 발음하는 영상을 한 번 찍은 뒤,
이 스크립트로 정규화된 입술 좌표 시퀀스만 뽑아 JSON으로 저장해둔다.
얼굴 전체 468개 랜드마크를 그대로 저장하면 파일이 불필요하게 커지므로,
compute_mouth_accuracy가 실제로 쓰는 입술 부분만 정규화까지 마쳐서 저장한다.

사용법:
    python scripts/extract_reference_landmarks.py <입력 영상 경로> <출력 JSON 경로>

예)
    python scripts/extract_reference_landmarks.py videos/annyeong.mp4 references/annyeong.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.mediapipe_service import extract_lip_sequence_from_video  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print("사용법: python scripts/extract_reference_landmarks.py <입력 영상 경로> <출력 JSON 경로>")
        raise SystemExit(1)

    video_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    video_bytes = video_path.read_bytes()
    sequence = extract_lip_sequence_from_video(video_bytes)

    if not sequence:
        print("영상에서 얼굴을 인식하지 못했습니다. 정면을 바라보는 영상인지, 조명이 밝은지 확인해주세요.")
        raise SystemExit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([frame.tolist() for frame in sequence]))
    print(f"{len(sequence)}개 프레임 추출 완료 -> {output_path}")


if __name__ == "__main__":
    main()
