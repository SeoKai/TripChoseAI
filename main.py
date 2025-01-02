# Flask 애플리케이션 생성 함수 임포트
from app import create_app, initialize_model
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

from app.config.app_config import MODEL_PATH
from app.models.trainer import load_data, get_max_ids, train_model, save_model

# Flask 애플리케이션 생성
app = create_app()


def update_model_periodically():
    """
    스케줄러를 통해 주기적으로 모델 업데이트 실행
    """
    with app.app_context():
        try:
            print("스케줄러에 의해 모델 업데이트 시작...")

            # 데이터베이스에서 데이터 로드
            ai_user_data, location_data = load_data()
            max_user_id, max_location_id = get_max_ids()

            # 데이터 전처리
            data = ai_user_data[['user_id', 'location_id', 'rating']]
            data.columns = ['user_id', 'place_id', 'rating']

            # 모델 학습
            print(f"데이터 전처리 완료: 사용자 ID 범위 [0, {max_user_id}], 장소 ID 범위 [0, {max_location_id}]")
            model = train_model(data, max_user_id, max_location_id)

            # 모델 저장
            save_model(model, MODEL_PATH)
            print("모델 업데이트 및 저장 완료.")
        except Exception as e:
            print(f"모델 업데이트 중 오류 발생: {e}")


if __name__ == "__main__":
    print("Flask 애플리케이션 초기화 중...")

    # 모델 초기 로드
    initialize_model()

    # 스케줄러 설정
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=update_model_periodically, trigger="interval", minutes=5)  # 5분마다
    scheduler.start()

    # 애플리케이션 종료 시 스케줄러 종료
    atexit.register(lambda: scheduler.shutdown())

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=5000, debug=True)
