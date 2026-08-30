"""발음 분석(STT + MediaPipe 통합) API."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db
from app.services import mediapipe_service, pitch_service, pronunciation_service, reference_service, stt_service

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

    # 억양(피치) 비교는 VIZ-001 음성 시각화용 참고 지표라 final_accuracy/is_correct
    # 판정에는 반영하지 않는다. 원어민 기준이 아직 등록되지 않았거나 발화에서
    # 유성 구간을 검출하지 못해도 STT/입모양 판정 자체는 그대로 진행되도록,
    # 실패해도 전체 요청을 실패시키지 않고 null로 남긴다.
    #
    # user 곡선은 reference 존재 여부와 무관하게 항상 내려준다 - 프론트가
    # VIZ-001-2(사용자 음성 실시간 추적)처럼 자신의 파형만 먼저 그려야 할 수도
    # 있고, reference가 없다는 이유로 방금 뽑은 사용자 데이터까지 버릴 이유는 없다.
    pitch_accuracy = None
    highlight_segments = None
    length_mismatch = False
    syllable_scores = None

    user_pitch = pitch_service.extract_pitch_sequence(audio_bytes)
    reference_pitch = reference_service.get_reference_pitch(db, target_text.strip()) if user_pitch else None

    if user_pitch and reference_pitch:
        try:
            # FUR-010: 유사도 점수 + 오차 구간 하이라이트
            comparison = pitch_service.compare_pitch_curves(
                user_pitch,
                reference_pitch,
                decay=settings.pitch_similarity_decay,
                error_threshold=settings.pitch_error_threshold_semitone,
                length_mismatch_ratio=settings.pitch_length_mismatch_ratio,
            )
            length_mismatch = comparison["length_mismatch"]
            if not length_mismatch:
                pitch_accuracy = comparison["similarity"]
                highlight_segments = comparison["highlight_segments"]

                # FUR-011: 음절별 구간 점수 (길이 차이가 과도해 비교 자체가 생략된 경우는 스킵)
                syllable_scores = pitch_service.score_syllables(
                    user_pitch,
                    reference_pitch,
                    target_text,
                    decay=settings.pitch_similarity_decay,
                    pass_threshold=settings.pass_threshold,
                )
        except (TypeError, KeyError):
            # DB에 저장된 reference_pitch가 예전 포맷(list[float], time 정보 없음)인 경우.
            # 억양 시각화는 참고 지표일 뿐이므로 이 경우에도 STT/입모양 판정은 막지 않고,
            # 피치 관련 필드만 비워서 응답한다. /reference를 다시 호출해 새 포맷으로
            # 재등록하면 해결된다.
            pitch_accuracy = None
            highlight_segments = None
            length_mismatch = False
            syllable_scores = None

    pitch_curve = {
        "user": user_pitch or None,
        "reference": reference_pitch,
        "highlight_segments": highlight_segments,
        "length_mismatch": length_mismatch,
    }

    return {
        "recognized_text": recognized_text,
        "stt_accuracy": round(stt_accuracy, 2),
        "mouth_accuracy": round(mouth_accuracy, 2) if mouth_accuracy is not None else None,
        "pitch_accuracy": round(pitch_accuracy, 2) if pitch_accuracy is not None else None,
        "pitch_curve": pitch_curve,
        "syllable_scores": syllable_scores,
        "final_accuracy": round(final_accuracy, 2),
        "is_correct": pronunciation_service.is_correct(final_accuracy, settings.pass_threshold),
    }


@router.post("/score")
def score_pronunciation(
    audio: UploadFile = File(..., description="학습자가 녹음한 발화 오디오 파일"),
    target_text: str = Form(..., description="학습자에게 제시된 단어/문장"),
):
    """게임용 - STT 정확도만 빠르게 산출한다 (억양·입모양 분석 생략).

    /analyze는 억양 곡선(피치 추출 + DTW 비교 + 음절 분할)까지 계산하는데,
    게임은 그중 final_accuracy 하나만 쓰고 나머지를 버린다. 영상을 받지 않으므로
    final_accuracy는 어차피 stt_accuracy와 같다(compute_final_accuracy 참고).
    버려질 계산을 애초에 하지 않도록 분리한 것이 이 엔드포인트다.

    STT도 게임용 경량 모델로 돌린다. 단어 하나의 정오답만 가리면 되므로
    발음 연습(/analyze)만큼의 정확도가 필요 없고, 게임은 응답 지연이 곧
    플레이 불가로 이어지기 때문이다.

    피드백 화면처럼 억양 곡선·음절별 점수가 필요한 경우에는 /analyze를 쓸 것.
    """
    audio_bytes = audio.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="빈 오디오 파일입니다.")

    try:
        recognized_text = stt_service.transcribe(audio_bytes, fast=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"오디오를 처리할 수 없습니다: {e}")

    stt_accuracy = pronunciation_service.compute_stt_accuracy(target_text, recognized_text)

    # 필드명은 /analyze와 맞춘다 - 스프링이 같은 파서로 읽을 수 있도록.
    return {
        "recognized_text": recognized_text,
        "stt_accuracy": round(stt_accuracy, 2),
        "final_accuracy": round(stt_accuracy, 2),
        "is_correct": pronunciation_service.is_correct(stt_accuracy, settings.pass_threshold),
    }


@router.get("/reference")
def get_reference(
    text: str = Query(..., description="조회할 학습 문장/단어 (analyze의 target_text와 동일해야 매칭됨)"),
    db: Session = Depends(get_db),
):
    """등록된 원어민 기준 발음의 억양(피치) 곡선을 조회한다 (VIZ-001-1 원어민 피치 가이드 조회).

    /reference(POST)는 저장 결과 개수만 돌려주므로, 학습자가 발음을 시도하기 전에
    프론트가 원어민 표준 피치 곡선을 먼저 그래프로 그려서 보여주려면 이 엔드포인트로
    실제 좌표 배열(pitch_curve)을 받아야 한다. text로 등록된 원어민 기준 자체가 없으면 404.
    pitch_curve는 오디오가 등록되지 않았거나 유성 구간을 검출하지 못했을 때 null일 수 있다.
    """
    landmarks = reference_service.get_reference_landmarks(db, text.strip())
    if landmarks is None:
        raise HTTPException(status_code=404, detail=f"'{text}'에 대해 등록된 원어민 기준 발음이 없습니다.")

    pitch = reference_service.get_reference_pitch(db, text.strip())

    return {
        "text": text.strip(),
        "landmark_frame_count": len(landmarks),
        "pitch_curve": pitch,
    }


@router.post("/reference")
def register_reference(
    text: str = Form(..., description="원어민 발음 영상/오디오에 대응하는 학습 문장/단어 (analyze의 target_text와 동일해야 매칭됨)"),
    video: UploadFile = File(..., description="원어민 표준 발음 웹캠/영상 파일 (입모양 랜드마크 추출용)"),
    audio: UploadFile | None = File(
        None, description="원어민 표준 발음 오디오 파일 (억양/피치 추출용, 영상과 별도 녹음 파일). 안 보내면 pitch는 저장되지 않는다."
    ),
    db: Session = Depends(get_db),
):
    """원어민 발음 영상·오디오에서 입모양 랜드마크와 억양(피치) 곡선을 추출해 text를 키로 DB에 저장(upsert)한다.

    문장/단어별로 1회만 호출해두면, 이후 /analyze 호출 시 같은 target_text로
    자동으로 이 기준 데이터를 찾아 비교한다. 이미 같은 text로 등록되어 있으면 덮어쓴다.

    입모양은 video에서, 억양(피치)은 별도 파일인 audio에서 추출한다 (영상 자체에 담긴
    소리는 쓰지 않음 - 프로젝트 구조상 음성과 영상이 별도로 녹음/업로드되기 때문).
    audio를 안 보내거나 오디오 처리에 실패해도(예: 손상된 파일) 입모양 기준 등록
    자체는 실패시키지 않고 pitch만 null로 남긴다 - 입모양 분석(필수 기능)이
    억양 시각화(참고 기능) 때문에 막히지 않도록 하기 위함이다.
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

    pitch = []
    if audio is not None:
        audio_bytes = audio.file.read()
        if audio_bytes:
            try:
                pitch = pitch_service.extract_pitch_sequence(audio_bytes)
            except ValueError:
                # 오디오 트랙이 없거나 손상된 파일 - 입모양 등록은 이미 성공했으니 억양만 비워둔다.
                pitch = []
    if pitch:
        saved = reference_service.save_reference_pitch(db, text.strip(), pitch)

    return {
        "text": saved.text,
        "frame_count": len(saved.landmarks),
        "pitch_frame_count": len(saved.pitch) if saved.pitch else 0,
    }
