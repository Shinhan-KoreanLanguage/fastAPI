"""문장/단어별 원어민 발음 랜드마크 저장·조회 서비스 (DB 접근 계층)."""
from sqlalchemy.orm import Session

from app.models import ReferencePronunciation


def get_reference_landmarks(db: Session, text: str) -> list | None:
    """text에 해당하는 원어민 랜드마크(원본 좌표)를 조회한다. 없으면 None."""
    row = db.query(ReferencePronunciation).filter_by(text=text).first()
    return row.landmarks if row else None


def save_reference_landmarks(db: Session, text: str, landmarks: list) -> ReferencePronunciation:
    """text를 키로 원어민 랜드마크를 저장한다. 이미 있으면 덮어쓴다(upsert)."""
    row = db.query(ReferencePronunciation).filter_by(text=text).first()
    if row is None:
        row = ReferencePronunciation(text=text, landmarks=landmarks)
        db.add(row)
    else:
        row.landmarks = landmarks

    db.commit()
    db.refresh(row)
    return row


def get_reference_pitch(db: Session, text: str) -> list | None:
    """text에 해당하는 원어민 억양(피치) 곡선을 조회한다. 없으면 None."""
    row = db.query(ReferencePronunciation).filter_by(text=text).first()
    return row.pitch if row else None


def save_reference_pitch(db: Session, text: str, pitch: list) -> ReferencePronunciation:
    """text를 키로 원어민 억양(피치) 곡선을 저장한다. 이미 있으면 덮어쓴다(upsert).

    save_reference_landmarks가 먼저 호출되어 text에 대한 행이 이미 존재하는
    상태(원어민 영상 등록 흐름)를 전제로 하므로, 새 행을 만들 때도 랜드마크가
    없는 반쪽 상태를 남기지 않도록 호출 순서에 주의해야 한다.
    """
    row = db.query(ReferencePronunciation).filter_by(text=text).first()
    if row is None:
        raise ValueError(f"'{text}'에 대한 원어민 랜드마크가 먼저 등록되어 있어야 합니다.")
    row.pitch = pitch

    db.commit()
    db.refresh(row)
    return row
