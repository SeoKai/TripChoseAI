import logging
from flask import Blueprint, request, jsonify
from app.config.db_config import load_data  # 데이터베이스 로드 함수
from app.models.trainer import load_model  # 모델 로드 함수
from app.models.recommender import recommend_places  # 추천 로직 함수
from app.config.app_config import MODEL_PATH

# Flask Blueprint 생성
bp = Blueprint('routes', __name__)

# 경로 확인 출력
print(f"Model Path: {MODEL_PATH}")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy Loading 설정
model = None
places = None
num_places = 0


class RecommenderService:
    """
    추천 시스템 상태 관리 클래스
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.places = []
        self.num_places = 0

    def initialize(self):
        """
        모델과 데이터를 초기화
        """
        try:
            logger.info("데이터베이스에서 데이터 로드 중...")
            ai_user_data, location_data = load_data()
            self.places = location_data['name'].tolist()
            self.num_places = len(self.places)

            logger.info(f"모델 로드 중: {self.model_path}")
            self.model = load_model(self.model_path)
            logger.info("모델 로드 성공!")
        except FileNotFoundError:
            logger.error("모델 파일이 없습니다. 학습을 먼저 진행하세요.")
        except Exception as e:
            logger.error(f"모델 초기화 중 오류 발생: {e}")
            self.model = None

    def recommend(self, user_id):
        """
        사용자 ID를 기반으로 추천 목록 반환
        """
        if self.model is None:
            raise RuntimeError("모델이 초기화되지 않았습니다. 학습 후 다시 시도하세요.")
        return recommend_places(self.model, user_id, self.num_places, self.places)


@bp.route('/recommend', methods=['POST'])
def recommend():
    # Content-Type 확인
    if not request.is_json:
        logger.warning("Content-Type 오류: JSON이 아님")
        return jsonify({"error": "Content-Type must be application/json"}), 415

    # 요청 데이터 확인
    data = request.get_json()
    logger.info(f"요청 받은 데이터: {data}")

    try:
        user_id = data.get("user_id")
        if user_id is None:
            logger.warning("user_id가 요청 데이터에 없음")
            return jsonify({"error": "user_id가 요청 데이터에 없습니다."}), 400

        # 추천 로직 실행
        recommendations = recommend_places(model, user_id, num_places, places)
        logger.info(f"추천 결과: {recommendations}")

        # JSON 배열로 반환
        return jsonify(recommendations), 200
    except Exception as e:
        logger.error(f"추천 API 처리 중 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500
