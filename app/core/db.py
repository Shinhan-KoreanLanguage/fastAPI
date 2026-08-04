"""MySQL 연결 설정 (SQLAlchemy).

원어민 발음 랜드마크처럼 DB에 영속시켜야 하는 데이터가 생기면 이 모듈의
`get_db`를 FastAPI Depends로 받아 세션을 사용하면 된다.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import settings

# pool_pre_ping: 커넥션 풀에 오래 idle 상태로 있던 커넥션이 MySQL 쪽에서 이미
# 끊겼는데 죽은 채로 재사용되는 것을 막기 위해, 매번 빌려주기 전에 살아있는지 확인한다.
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Session:
    """요청 하나당 세션 하나를 열고, 응답이 끝나면 반드시 닫아주는 FastAPI 의존성."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
