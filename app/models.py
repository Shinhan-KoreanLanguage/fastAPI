"""DB 테이블(ORM 모델) 정의."""
from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from app.core.db import Base


class ReferencePronunciation(Base):
    """문장/단어 텍스트별 원어민 표준 발음 입모양 랜드마크 및 억양(피치) 곡선.

    landmarks는 mediapipe_service.extract_raw_landmarks_from_video()가 반환하는
    프레임별 468개 [x, y] 좌표 배열(정규화 이전 원본)을 그대로 저장한다.
    비교 시점에 parse_landmark_sequence()로 정규화해 사용하므로, 정규화 로직이
    바뀌어도 DB에 저장된 원본 좌표는 다시 추출할 필요가 없다.

    pitch는 pitch_service.extract_pitch_sequence()가 반환하는 세미톤 정규화된
    피치 시계열이다. 원어민 영상에 오디오 트랙이 없거나 유성 구간을 검출하지
    못하면 null로 남을 수 있으므로(VIZ-001 억양 시각화는 참고 지표라 필수는
    아님) landmarks와 달리 nullable로 둔다.
    """

    __tablename__ = "reference_pronunciations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String(255), unique=True, nullable=False, index=True)
    landmarks = Column(JSON, nullable=False)
    pitch = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
