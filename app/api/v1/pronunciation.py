"""발음 분석(STT + MediaPipe 통합) API."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db
from app.services import mediapipe_service, pronunciation_service, reference_service, stt_service

# 이 라우터의 모든 경로 앞에 /pronunciation 이 붙는다 (예: /api/v1/pronunciation/analyze)
router = APIRouter(prefix="/pronunciation", tags=["발음 분석"])


@router.post("/analyze")
def analyze_pronunciation(
    audio: UploadFile = File(..., description="학습자가 녹음한 발화 오디오 파일"),
    target_text: str = Form(..., description="학습자에게 제시된 학습 문장/단어 (원어민 기준 데이터 조회 키로도 쓰임)"),
    video: UploadFile | None = File(
        None, description="학습자의 입모양이 담긴 웹캠 녹화 영상 (DB에 target_text로 등록된 원어민 기준이 있어야 입모양 정확도가 산출됨)"
    ),
    db: Session = Depends(get_db),
):
    """녹음된 발화 오디오와 입모양 영상을 함께 분석해 통합 발음 정확도를 산출한다.

    원어민 기준 좌표는 매번 영상으로 받지 않고, target_text를 키로 DB에서 조회한다
    (사전에 /reference 로 등록되어 있어야 함). video를 보냈는데 DB에 target_text로
    등록된 기준이 없으면 404를 반환한다. video 자체를 안 보내면 STT 결과만으로 정확도를 산출한다.
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
    if video is not None:
        raw_reference_landmarks = reference_service.get_reference_landmarks(db, target_text.strip())
        if raw_reference_landmarks is None:
            raise HTTPException(
                status_code=404,
                detail=f"'{target_text}'에 대해 등록된 원어민 기준 발음이 없습니다. 먼저 /reference 로 등록해주세요.",
            )
        try:
            user_sequence = mediapipe_service.extract_lip_sequence_from_video(video.file.read())
            reference_sequence = mediapipe_service.parse_landmark_sequence(raw_reference_landmarks)
            if user_sequence and reference_sequence:
                mouth_accuracy = mediapipe_service.compute_mouth_accuracy(
                    user_sequence, reference_sequence, decay=settings.mouth_accuracy_decay
                )
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
        "is_correct": pronunciation_service.is_correct(final_accuracy, settings.pass_threshold),
    }


@router.post("/reference")
def register_reference(
    text: str = Form(..., description="원어민 발음 영상에 대응하는 학습 문장/단어 (analyze의 target_text와 동일해야 매칭됨)"),
    video: UploadFile = File(..., description="원어민 표준 발음 웹캠/영상 파일"),
    db: Session = Depends(get_db),
):
    """원어민 발음 영상에서 입모양 랜드마크를 추출해 text를 키로 DB에 저장(upsert)한다.

    문장/단어별로 1회만 호출해두면, 이후 /analyze 호출 시 같은 target_text로
    자동으로 이 기준 데이터를 찾아 비교한다. 이미 같은 text로 등록되어 있으면 덮어쓴다.
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

    saved = reference_service.save_reference_landmarks(db, text.strip(), landmarks)
    return {"text": saved.text, "frame_count": len(saved.landmarks)}
