# Flask 관련 모듈 임포트
from flask import Flask

# Blueprint 모듈 임포트
from app.routes import bp, initialize_model


# Flask 애플리케이션 생성 함수 정의
def create_app():
    # Flask 애플리케이션 인스턴스 생성
    app = Flask(__name__)

    # Blueprint 등록
    app.register_blueprint(bp)

    initialize_model()

    # 생성된 애플리케이션 반환
    return app
