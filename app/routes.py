import os
from flask import Blueprint, request, jsonify
from app.config.db_config import load_data  # 데이터베이스 로드 함수
from app.models.trainer import load_model  # 모델 로드 함수
from app.models.recommender import recommend_places  # 추천 로직 함수

# Flask Blueprint 생성
bp = Blueprint('routes', __name__)

# 모델 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR,"app", "models", "saved_model", "recommender_model.keras")

# 경로 확인 출력
print(f"Model Path: {model_path}")

# Lazy Loading 설정
model = None
places = None
num_places = 0


def initialize_model():
    """
    모델과 데이터를 지연 로드
    """
    global model, places, num_places

    if model is None:
        print("데이터베이스에서 데이터 로드 중...")
        ai_user_data, location_data = load_data()
        places = location_data['name'].tolist()  # 여행지 이름 리스트
        num_places = len(places)  # 여행지 총 개수

        print(f"{model_path}에서 모델을 로드합니다...")
        try:
            model = load_model(model_path)
            print("모델 로드 성공!")
        except FileNotFoundError:
            print("모델 파일이 없습니다. API 호출 전 학습을 진행하세요.")
            model = None
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")  # 예외 메시지 출력
            model = None


@bp.route('/recommend', methods=['POST'])
def recommend():
    # Content-Type 확인
    if not request.is_json:
        print("Content-Type 오류: JSON이 아님")
        return jsonify({"error": "Content-Type must be application/json"}), 415

    # 요청 데이터 확인
    data = request.get_json()
    print("받은 데이터:", data)

    try:
        user_id = data.get("user_id")
        if user_id is None:
            print("user_id가 요청 데이터에 없음")
            return jsonify({"error": "Missing user_id in request"}), 400

        # 추천 로직 실행
        recommendations = recommend_places(model, user_id, num_places, places)
        print("추천 결과:", recommendations)

        # JSON 배열로 반환
        return jsonify(recommendations), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
