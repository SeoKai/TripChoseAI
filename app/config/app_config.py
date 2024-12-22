import os

# 프로젝트 루트 기준으로 상대경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # `app_config.py` 파일 위치 기준
MODEL_DIR = os.path.join(BASE_DIR, "../models/saved_model")  # 상대경로 설정
MODEL_PATH = os.path.join(MODEL_DIR, "recommender_model.keras")
