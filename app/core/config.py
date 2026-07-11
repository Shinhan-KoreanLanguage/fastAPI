"""환경설정. .env 파일의 값을 읽어 들입니다.

스프링에서 application.yml 로 설정을 관리하듯, 파이썬에서는 보통
pydantic-settings 로 .env 값을 읽어 이 Settings 객체 하나로 관리합니다.
기능별 설정값(예: 게임 점수 가중치 등)은 각 기능을 만들 때 여기에 추가하세요.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 앱 기본 정보
    app_name: str = "Pronunciation AI Service"
    api_v1_prefix: str = "/api/v1"

    # CORS 허용 출처. 스프링 서버 주소를 넣으세요. 여러 개면 콤마로 구분.
    #   예) CORS_ORIGINS=http://localhost:8080,http://localhost:3000
    cors_origins: str = "*"

    # 스프링부트만 이 서버를 호출하도록 막을 때 쓰는 공유 키 (선택, 비우면 검사 안 함)
    service_api_key: str | None = None

    # 통합 발음 정확도 가중치 (STT 정확도 + MediaPipe 입모양 정확도 = 1.0).
    # 실측 데이터에 따라 조정 가능하도록 설정값으로 분리 (기본값: STT 0.7 + MediaPipe 0.3)
    stt_weight: float = 0.7
    mediapipe_weight: float = 0.3

    # 입모양 정확도 채점 민감도 (지수 감쇠 상수). 값이 클수록 같은 오차에도
    # 점수가 더 가파르게 떨어진다. 채점이 너무 박하면 이 값을 낮추세요.
    mouth_accuracy_decay: float = 3.0

    # 통합 발음 정확도가 이 값 이상이면 "맞은 것"으로 판정 (게임 점수 합격선)
    pass_threshold: float = 85.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# 앱 전체에서 이 settings 하나를 import 해서 사용합니다.
settings = Settings()
