import os

# 프로젝트 루트 기준으로 상대경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # `app_config.py` 파일 위치 기준
MODEL_DIR = os.path.join(BASE_DIR, "../models/saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "recommender_model.keras")
LOG_PATH = os.path.join(BASE_DIR, "../logs")  # 로깅 디렉토리

# 모델 경로 초기화 함수
def initialize_model_paths():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    print("모델 및 로그 경로 초기화 완료")