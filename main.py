import logging
# Flask 애플리케이션 생성 함수 임포트
from app import create_app

# Flask 애플리케이션 인스턴스 생성
app = create_app()

# 로깅 설정
logging.basicConfig(filename='/home/ec2-user/flask-project/flask.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

@app.route('/')
def hello_world():
    app.logger.info('Hello World endpoint was hit.')
    return 'Hello, World!'

# Python 스크립트가 직접 실행될 경우 Flask 개발 서버 시작
if __name__ == "__main__":
    # 디버그 모드로 Flask 서버 실행
    app.run(host="0.0.0.0", port=5000, debug=True)
