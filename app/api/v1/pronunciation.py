"""발음 분석(STT + MediaPipe 통합) API."""
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.services import mediapipe_service, pronunciation_service, stt_service

# 이 라우터의 모든 경로 앞에 /pronunciation 이 붙는다 (예: /api/v1/pronunciation/analyze)
router = APIRouter(prefix="/pronunciation", tags=["발음 분석"])


@router.post("/analyze")
def analyze_pronunciation(
    audio: UploadFile = File(..., description="학습자가 녹음한 발화 오디오 파일"),
    target_text: str = Form(..., description="학습자에게 제시된 학습 문장"),
    landmarks: str | None = Form(
        None, description="클라이언트(MediaPipe.js)가 추출한 프레임별 468개 랜드마크 좌표 (JSON 배열)"
    ),
    reference_landmarks: str | None = Form(
        None, description="원어민 표준 발음의 프레임별 468개 랜드마크 좌표 (JSON 배열)"
    ),
    video: UploadFile | None = File(
        None, description="mediapipe로 서버에서 직접 랜드마크를 추출할 웹캠 녹화 영상 (landmarks 미제공 시 사용)"
    ),
):
    """녹음된 발화 오디오와 입모양 데이터를 함께 분석해 통합 발음 정확도를 산출한다.

    입모양 데이터는 (1) 클라이언트가 이미 추출한 랜드마크 좌표(landmarks) 또는
    (2) 서버에서 mediapipe로 직접 분석할 영상 파일(video) 중 하나로 받을 수 있다.
    reference_landmarks 가 없거나 사용자 입모양 데이터가 없으면 STT 결과만으로 정확도를 산출한다.
    """
    audio_bytes = audio.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="빈 오디오 파일입니다.")

    try:
        recognized_text = stt_service.transcribe(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"오디오를 처리할 수 없습니다: {e}")

    stt_accuracy = pronunciation_service.compute_stt_accuracy(target_text, recognized_text)

    mouth_accuracy = None
    if reference_landmarks:
        try:
            reference_sequence = mediapipe_service.parse_landmark_sequence(json.loads(reference_landmarks))

            if landmarks:
                user_sequence = mediapipe_service.parse_landmark_sequence(json.loads(landmarks))
            elif video is not None:
                user_sequence = mediapipe_service.extract_lip_sequence_from_video(video.file.read())
            else:
                user_sequence = None

            if user_sequence:
                mouth_accuracy = mediapipe_service.compute_mouth_accuracy(user_sequence, reference_sequence)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=422, detail=f"랜드마크 데이터가 올바른 JSON이 아닙니다: {e}")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"입모양 데이터를 처리할 수 없습니다: {e}")

    final_accuracy = pronunciation_service.compute_final_accuracy(
        stt_accuracy, mouth_accuracy, settings.stt_weight, settings.mediapipe_weight
    )

    return {
        "recognized_text": recognized_text,
        "stt_accuracy": round(stt_accuracy, 2),
        "mouth_accuracy": round(mouth_accuracy, 2) if mouth_accuracy is not None else None,
        "final_accuracy": round(final_accuracy, 2),
    }
