# Flask 관련 모듈 및 외부 의존성 임포트
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# 내부 모듈 임포트
from app.routes import bp, RecommenderService
from app.config.app_config import MODEL_PATH
from app.models.trainer import load_data, get_max_ids, train_model, save_model

# 추천 서비스 인스턴스 생성
recommender_service = RecommenderService(MODEL_PATH)

def create_app():
    """
    Flask 애플리케이션 생성 및 설정
    """
    app = Flask(__name__)

    # Blueprint 등록
    app.register_blueprint(bp)

    # 모델 초기화 (routes 내부에서 수행)
    recommender_service.initialize()

    return app

def update_model_periodically(app):
    """
    스케줄러를 통해 주기적으로 모델 업데이트 실행
    """
    with app.app_context():
        try:
            print("스케줄러에 의해 모델 업데이트 시작...")
            ai_user_data, location_data = load_data()
            max_user_id, max_location_id = get_max_ids()

            # 데이터 전처리
            data = ai_user_data[['user_id', 'location_id', 'rating']]
            data.columns = ['user_id', 'place_id', 'rating']

            # 모델 학습
            model = train_model(data, max_user_id, max_location_id)

            # 모델 저장
            save_model(model, MODEL_PATH)
            print("모델 업데이트 및 저장 완료.")

            # RecommenderService에서 모델 재로드
            recommender_service.reload_model()
            print("RecommenderService에서 모델을 재로드했습니다.")

        except Exception as e:
            print(f"모델 업데이트 중 오류 발생: {e}")


if __name__ == "__main__":
    print("Flask 애플리케이션 초기화 중...")

    # Flask 애플리케이션 생성
    app = create_app()

    # 스케줄러 설정
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=lambda: update_model_periodically(app), trigger="interval", seconds=10)
    scheduler.start()

    # 애플리케이션 종료 시 스케줄러 종료
    atexit.register(lambda: scheduler.shutdown())

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=5000, debug=False)
