"""FastAPI 앱 시작점 (공통 뼈대).

실행:  uvicorn app.main:app --reload
문서:  http://127.0.0.1:8000/docs  (Swagger - 만든 API 를 여기서 바로 테스트)

기능별 라우터는 app/api/v1/ 안에 파일을 만든 뒤, 아래 '라우터 등록' 부분에
한 줄씩 추가하면 됩니다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import pronunciation, stt
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="MediaPipe·STT 기반 한국어 발음 교정 - AI 분석 전용 서버",
    version="1.0.0",
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
