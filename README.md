# Pronunciation AI Service

MediaPipe·STT 기반 외국인 한국어 발음 교정 서비스의 **AI 분석 전용 서버**입니다.
FastAPI 로 만들며, 오디오·영상을 받아 분석 결과(JSON)만 돌려주는 무상태 서버입니다.

회원/콘텐츠/랭킹/통계 등 일반 CRUD 와 MySQL 저장은 **스프링부트**가 담당합니다.

```
[앱] → [Spring Boot : CRUD·인증·MySQL] → [FastAPI(이 서버) : AI 분석]
```

## 폴더 구조

```
pronunciation-ai/
├── app/
│   ├── main.py          # 앱 시작점. CORS 설정, /health, 라우터 등록
│   ├── core/
│   │   └── config.py    # 환경설정(.env 값을 읽는 곳)
│   ├── api/
│   │   └── v1/          # 기능별 API(라우터) 파일을 추가하는 곳
│   └── services/        # 실제 로직·AI 분석 코드를 작성하는 곳
├── requirements.txt     # 설치할 패키지 목록
├── .env.example         # 환경설정 예시 (복사해서 .env 로 사용)
└── .gitignore
```

| 위치 | 역할 | 스프링에 비유하면 |
|------|------|-----------------|
| `app/api/v1/` | API 주소(URL) 정의 | Controller |
| `app/services/` | 비즈니스 로직 / AI 연동 | Service |
| `app/core/config.py` | 환경설정 | application.yml |

## 주요 기능 (예정)

아직 뼈대만 있는 상태이며, 아래 기능을 하나씩 추가해 나갑니다.

| 기능 | 설명 |
|------|------|
| 발음 분석 | 사용자 음성을 STT 로 변환하고 발음 정확도·교정 피드백 산출 |
| 음성 시각화 | 음성에서 피치(음높이)를 추출하고 원어민 발음과 비교(DTW) |
| 입모양 분석 | MediaPipe 로 입모양을 분석해 발음 보조 점수 산출 |
| 게임 점수 판정 | 음성·입모양 점수를 합산해 합격 여부 판정 |

## 실행 방법 (Windows / PowerShell)

```powershell
cd C:\Users\socce\pronunciation-ai

python -m venv .venv               # 가상환경 생성 (최초 1회)
.\.venv\Scripts\Activate.ps1       # 가상환경 활성화
pip install -r requirements.txt    # 패키지 설치
copy .env.example .env             # 환경설정 파일 생성

uvicorn app.main:app --reload      # 서버 실행
```

실행 후 브라우저에서 **http://127.0.0.1:8000/docs** 를 열면
만든 API 를 화면에서 바로 테스트할 수 있습니다. (현재는 `/health` 만 있음)
