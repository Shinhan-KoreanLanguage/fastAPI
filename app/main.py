"""FastAPI 앱 시작점 (공통 뼈대).

실행:  uvicorn app.main:app --reload   (또는  python app/main.py)
문서:  http://127.0.0.1:8000/docs  (Swagger - 만든 API 를 여기서 바로 테스트)

기능별 라우터는 app/api/v1/ 안에 파일을 만든 뒤, 아래 '라우터 등록' 부분에
한 줄씩 추가하면 됩니다.
"""
import sys
from pathlib import Path

# `python app/main.py` 처럼 직접 실행하면 sys.path[0]이 이 파일이 있는 app/ 폴더가 되어
# 프로젝트 루트 기준의 `app.xxx` 절대 임포트를 찾지 못한다. 프로젝트 루트를 앞에 넣어 보정.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import pronunciation, stt
from app.core.config import settings
from app.core.db import Base, engine
from app.services import stt_service
from app import models  # noqa: F401  (Base.metadata에 테이블을 등록시키기 위한 임포트)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 서버 시작 시 없는 테이블만 생성 (이미 있으면 건드리지 않음).
    # MySQL이 아직 안 떠 있어도 STT 등 DB와 무관한 기능은 계속 쓸 수 있도록,
    # 실패해도 앱 기동 자체는 막지 않고 경고만 남긴다.
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[경고] DB 테이블 생성 실패 (MySQL 접속 설정을 확인하세요): {e}")

    # STT 모델을 미리 로드해둔다. lru_cache 특성상 그냥 두면 프로세스의 첫 요청이
    # 모델 로딩 시간까지 떠안는데(실측 base 약 1.2초, small 약 2.7초), 하필 그 첫
    # 요청이 게임 중인 사용자일 수 있다. 기동이 조금 늦어지더라도 요청 지연을
    # 일정하게 만드는 편이 낫다.
    # 모델 파일이 아직 없으면 최초 1회 다운로드가 일어나므로 여기서 더 오래 걸린다.
    # 네트워크 문제로 실패해도 서버 기동 자체는 막지 않는다 - 그 경우 예전처럼
    # 첫 요청 때 다시 시도하게 된다.
    for model_name in (stt_service.GAME_MODEL, stt_service.DETAIL_MODEL):
        try:
            stt_service._load_model(model_name)
        except Exception as e:
            print(f"[경고] STT 모델 사전 로드 실패 ({model_name}): {e}")

    yield


app = FastAPI(
    title=settings.app_name,
    description="MediaPipe·STT 기반 한국어 발음 교정 - AI 분석 전용 서버",
    version="1.0.0",
    lifespan=lifespan,
)

# 스프링부트(다른 포트)에서 호출할 수 있도록 CORS 허용.
_origins = (
    ["*"]
    if settings.cors_origins.strip() == "*"
    else [o.strip() for o in settings.cors_origins.split(",")]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["기본"])
def health_check():
    """서버가 살아있는지 확인용."""
    return {"status": "ok", "service": settings.app_name}


# ──────────────────────────────────────────────────────────────
# 라우터 등록 — 기능 파일을 만들면 여기에 한 줄씩 추가하세요.
#
# 예) app/api/v1/pronunciation.py 안에 router = APIRouter(...) 를 만든 뒤:
#
#   from app.api.v1 import pronunciation
#   app.include_router(pronunciation.router, prefix=settings.api_v1_prefix)
# ──────────────────────────────────────────────────────────────
app.include_router(stt.router, prefix=settings.api_v1_prefix)
app.include_router(pronunciation.router, prefix=settings.api_v1_prefix)


if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0": 루프백(127.0.0.1)뿐 아니라 이 PC의 모든 네트워크 인터페이스에서
    # 들어오는 연결을 받는다. 공유기 포트포워딩으로 외부 접속을 받으려면 필수.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
