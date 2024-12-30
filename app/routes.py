import os
from flask import Blueprint, request, jsonify
from app.config.db_config import load_data  # 데이터베이스 로드 함수
from app.models.trainer import load_model  # 모델 로드 함수
from app.models.recommender import recommend_places  # 추천 로직 함수

# Flask Blueprint 생성
bp = Blueprint('routes', __name__)

# 모델 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "app", "models", "saved_model", "recommender_model.keras")

# 경로 확인 출력
print(f"Model Path: {model_path}")

# 데이터 로드
print("데이터베이스에서 데이터 로드 중...")
ai_user_data, location_data = load_data()

# 장소 데이터 가공
places = location_data['name'].tolist()  # 여행지 이름 리스트
num_places = len(places)  # 여행지 총 개수

# 모델 로드
try:
    model = load_model(model_path)
    print("모델 로드 성공!")
except FileNotFoundError:
    print("모델 파일이 없습니다. API 호출 전 학습을 진행하세요.")
    model = None


@bp.route('/recommend', methods=['POST'])
def recommend():
    """
    추천 API 엔드포인트
    """
    try:
        # Content-Type 확인
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        # 클라이언트로부터 받은 JSON 데이터 파싱
        data = request.get_json()
        user_id = data.get("user_id")

        # user_id가 포함되어 있는지 확인
        if user_id is None:
            return jsonify({"error": "Missing user_id in request"}), 400

        # 모델이 로드되지 않은 경우 에러 반환
        if model is None:
            return jsonify({"error": "Model not loaded. Train the model first."}), 500

        # user_id가 모델의 범위 내에 있는지 확인
        if user_id < 0 or user_id >= model.num_users:
            return jsonify({"error": f"user_id must be between 0 and {model.num_users - 1}"}), 400

        # 추천 결과 생성
        recommendations = recommend_places(model, user_id, num_places=5, places=places)

        # 추천 결과를 JSON 형태로 반환
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        # 예외 처리 및 에러 메시지 반환
        print(f"Error: {e}")  # 로그로 에러 출력
        return jsonify({"error": "Internal server error"}), 500
