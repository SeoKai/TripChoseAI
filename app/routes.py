# Flask 관련 모듈 임포트
from flask import Blueprint, request, jsonify

# 추천 시스템 및 모델 관련 모듈 임포트
from app.models.recommender import recommend_places
from app.models.trainer import load_model

# 데이터 처리 및 파일 경로 관련 모듈 임포트
import pandas as pd

# 모델 로드
model_path = "C:/TripChose/app/models/saved_model/recommender_model.keras"
model = load_model(model_path)

# 여행지 데이터 로드
data_path = "C:/TripChose/data/places.csv"
places_df = pd.read_csv(data_path)

# 전역 변수 정의
num_places = len(places_df)  # 여행지 총 개수
places = places_df['name'].tolist()  # 여행지 이름 리스트

# Flask Blueprint 생성
bp = Blueprint('routes', __name__)


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
