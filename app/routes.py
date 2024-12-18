import os
import pandas as pd
from flask import Blueprint, request, jsonify
from app.models.recommender import recommend_places
from app.models.trainer import load_model

# 현재 파일을 기준으로 프로젝트 루트 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 모델 경로와 데이터 경로 설정
model_path = os.path.join(BASE_DIR, "app", "models", "saved_model", "recommender_model.keras")
data_path = os.path.join(BASE_DIR, "data", "places.csv")

# 경로 확인 출력
print(f"Model Path: {model_path}")
print(f"Data Path: {data_path}")

# 모델 및 데이터 로드
model = load_model(model_path)
places_df = pd.read_csv(data_path)

# Flask Blueprint 생성
bp = Blueprint('routes', __name__)


# 전역 변수 정의
num_places = len(places_df)  # 여행지 총 개수
places = places_df['name'].tolist()  # 여행지 이름 리스트



# 추천 API 엔드포인트 정의
@bp.route('/recommend', methods=['POST'])
def recommend():
    try:
        # 클라이언트로부터 받은 JSON 데이터 파싱
        data = request.get_json()
        user_id = data.get("user_id")

        # user_id가 포함되어 있는지 확인
        if user_id is None:
            return jsonify({"error": "user_id를 포함해야 합니다."}), 400

        # user_id가 모델의 범위 내에 있는지 확인
        if user_id < 0 or user_id >= model.num_users:
            return jsonify({"error": f"user_id는 0에서 {model.num_users - 1} 사이여야 합니다."}), 400

        # 추천 결과 생성
        recommendations = recommend_places(model, user_id, num_places=5, places=places)

        # 추천 결과를 JSON 형태로 반환
        return jsonify({"recommendations": recommendations})

    # 예외 처리 및 에러 메시지 반환
    except Exception as e:
        return jsonify({"error": str(e)}), 500
