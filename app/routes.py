import logging
from flask import Blueprint, request, jsonify
from app.config.db_config import load_data, get_max_ids  # 데이터베이스 로드 함수
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

        # 사용자 ID 검증
        if user_id < 0 or user_id >= self.num_users:
            raise ValueError(f"user_id {user_id}는 모델의 사용자 범위를 초과합니다. (모델 사용자 수: {self.num_users})")

        # 추천 로직 실행
        return recommend_places(self.model, user_id, self.num_places, self.places)

    def reload_model(self):
        """
        저장된 모델을 다시 로드하고, num_users와 num_places를 업데이트합니다.
        """
        try:
            logger.info(f"모델 재로드 중: {self.model_path}")
            self.model = load_model(self.model_path)

            # 데이터베이스에서 최신 user_id 및 place_id 가져오기
            max_user_id, max_place_id = get_max_ids()
            self.num_users = max_user_id + 1
            self.num_places = max_place_id + 1

            logger.info(f"모델 재로드 성공! num_users={self.num_users}, num_places={self.num_places}")
        except Exception as e:
            logger.error(f"모델 재로드 중 오류 발생: {e}")


# RecommenderService 인스턴스 생성 및 초기화
service = RecommenderService(model_path=MODEL_PATH)
service.initialize()  # 앱 시작 시 모델과 데이터 초기화

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

        # 최신 모델 로드
        service.reload_model()

        # RecommenderService 인스턴스를 통해 추천 로직 실행
        recommendations = service.recommend(user_id)
        logger.info(f"추천 결과: {recommendations}")

        # JSON 배열로 반환
        return jsonify(recommendations), 200
    except RuntimeError as e:
        logger.error(f"추천 API 처리 중 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"추천 API 처리 중 예기치 못한 오류 발생: {e}")
        return jsonify({"error": "알 수 없는 오류가 발생했습니다."}), 500
