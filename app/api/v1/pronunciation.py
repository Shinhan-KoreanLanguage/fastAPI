"""발음 분석(STT + MediaPipe 통합) API."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.services import mediapipe_service, pronunciation_service, stt_service

# 이 라우터의 모든 경로 앞에 /pronunciation 이 붙는다 (예: /api/v1/pronunciation/analyze)
router = APIRouter(prefix="/pronunciation", tags=["발음 분석"])


@router.post("/analyze")
def analyze_pronunciation(
    audio: UploadFile = File(..., description="학습자가 녹음한 발화 오디오 파일"),
    target_text: str = Form(..., description="학습자에게 제시된 학습 문장"),
    video: UploadFile | None = File(
        None, description="학습자의 입모양이 담긴 웹캠 녹화 영상 (reference_video와 함께 보내야 입모양 정확도가 산출됨)"
    ),
    reference_video: UploadFile | None = File(
        None, description="원어민 표준 발음 웹캠/영상 파일 (video와 함께 보내야 입모양 정확도가 산출됨)"
    ),
):
    """녹음된 발화 오디오와 입모양 영상을 함께 분석해 통합 발음 정확도를 산출한다.

    video와 reference_video를 모두 보내면 서버에서 mediapipe로 각 영상의 랜드마크를
    직접 추출해 입모양 정확도를 함께 산출한다. 랜드마크 좌표를 JSON으로 직접 보내지
    않아도 되도록, 두 영상 모두 멀티파트 파일로 받는다. 둘 다 없거나 하나만 있으면
    STT 결과만으로 정확도를 산출한다.
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
    if video is not None and reference_video is not None:
        try:
            user_sequence = mediapipe_service.extract_lip_sequence_from_video(video.file.read())
            reference_sequence = mediapipe_service.extract_lip_sequence_from_video(reference_video.file.read())
            if user_sequence and reference_sequence:
                mouth_accuracy = mediapipe_service.compute_mouth_accuracy(
                    user_sequence, reference_sequence, decay=settings.mouth_accuracy_decay
                )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"입모양 데이터를 처리할 수 없습니다: {e}")
    elif video is not None or reference_video is not None:
        raise HTTPException(
            status_code=400, detail="입모양 정확도를 산출하려면 video와 reference_video를 모두 보내야 합니다."
        )

    final_accuracy = pronunciation_service.compute_final_accuracy(
        stt_accuracy, mouth_accuracy, settings.stt_weight, settings.mediapipe_weight
    )

    return {
        "recognized_text": recognized_text,
        "stt_accuracy": round(stt_accuracy, 2),
        "mouth_accuracy": round(mouth_accuracy, 2) if mouth_accuracy is not None else None,
        "final_accuracy": round(final_accuracy, 2),
        "is_correct": pronunciation_service.is_correct(final_accuracy, settings.pass_threshold),
    }


@router.post("/reference-landmarks")
def extract_reference_landmarks(
    video: UploadFile = File(..., description="원어민 표준 발음 웹캠/영상 파일"),
):
    """원어민 발음 영상에서 랜드마크를 추출해 반환한다.

    /analyze는 영상을 매 요청마다 직접 받아 처리하므로 이 엔드포인트가 필수는 아니며,
    추출된 랜드마크 좌표를 확인·디버깅하고 싶을 때 사용한다.
    """
    video_bytes = video.file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="빈 영상 파일입니다.")

    try:
        landmarks = mediapipe_service.extract_raw_landmarks_from_video(video_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"영상을 처리할 수 없습니다: {e}")

    if not landmarks:
        raise HTTPException(status_code=422, detail="영상에서 얼굴을 인식하지 못했습니다.")

    return {"landmarks": landmarks}
