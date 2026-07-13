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
