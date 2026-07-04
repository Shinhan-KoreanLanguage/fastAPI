"""STT(음성 -> 텍스트) API."""
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services import stt_service

# 이 라우터의 모든 경로 앞에 /stt 가 붙는다 (예: /api/v1/stt/transcribe)
router = APIRouter(prefix="/stt", tags=["STT"])


@router.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...)):
    """업로드된 오디오 파일을 Whisper로 텍스트 변환하여 반환한다.

    STT 추론은 CPU를 오래 점유하는 동기 작업이므로, 일부러 async def가 아닌
    일반 def로 선언한다. FastAPI는 sync 엔드포인트를 스레드풀에서 실행해
    이벤트 루프가 추론 중 다른 요청(헬스체크 등)까지 막는 것을 방지한다.
    """
    # 업로드된 파일 내용을 바이트로 읽어들임 (sync 엔드포인트이므로 await 불필요)
    audio_bytes = file.file.read()
    if not audio_bytes:
        # 파일이 비어있으면 서버 로직을 태우지 않고 바로 400 에러 응답
        raise HTTPException(status_code=400, detail="빈 오디오 파일입니다.")

    try:
        # 실제 STT 변환은 서비스 계층(stt_service)에 위임
        text = stt_service.transcribe(audio_bytes)
    except Exception as e:
        # 오디오 디코딩 실패 등 처리 중 에러가 나면 422(처리 불가)로 응답
        raise HTTPException(status_code=422, detail=f"오디오를 처리할 수 없습니다: {e}")

    return {"text": text}
