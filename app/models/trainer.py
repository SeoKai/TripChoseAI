from app.config.app_config import MODEL_DIR, MODEL_PATH
import os
import tensorflow as tf
from app.config.db_config import load_data, get_max_ids


class RecommenderModel(tf.keras.Model):
    """
    추천 시스템 모델 정의.
    사용자, 장소의 임베딩 생성
    내적(dot)을 계산하여 선호도 예측
    """
    def __init__(self, num_users, num_places, embedding_dim=50, **kwargs):
        super(RecommenderModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_dim = embedding_dim
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)  # 사용자 임베딩
        self.place_embedding = tf.keras.layers.Embedding(num_places, embedding_dim)  # 장소 임베딩
        self.dot = tf.keras.layers.Dot(axes=1)  # 사용자와 장소 임베딩의 내적 계산

    def call(self, inputs):
        if isinstance(inputs, tuple):
            user_ids, place_ids = inputs
        else:
            user_ids, place_ids = inputs[:, 0], inputs[:, 1]

        user_vector = self.user_embedding(user_ids)
        place_vector = self.place_embedding(place_ids)
        return self.dot([user_vector, place_vector])

    def get_config(self):
        config = super(RecommenderModel, self).get_config()
        config.update({
            "num_users": self.num_users,
            "num_places": self.num_places,
            "embedding_dim": self.embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        num_users = config.pop("num_users")
        num_places = config.pop("num_places")
        embedding_dim = config.pop("embedding_dim")
        return cls(num_users=num_users, num_places=num_places, embedding_dim=embedding_dim, **config)


def train_model(train_data, max_user_id, max_location_id):
    """
    추천 모델 학습

    Args:
        train_data (pd.DataFrame): 학습 데이터 (user_id, place_id, rating)
        max_user_id (int): 최대 user_id 값
        max_location_id (int): 최대 location_id 값

    Returns:
        RecommenderModel: 학습된 모델
    """
    X_train = train_data[['user_id', 'place_id']].values
    y_train = train_data['rating'].values

    # 모델 초기화
    num_users = max_user_id + 1  # user_id가 0부터 시작하므로 +1
    num_places = max_location_id + 1  # place_id가 0부터 시작하므로 +1

    print(f"임베딩 크기 - 사용자: {num_users}, 장소: {num_places}")
    model = RecommenderModel(num_users, num_places)

    print("모델 구조:")
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    # 모델 학습
    print("모델 학습 시작")
    model.fit((X_train[:, 0], X_train[:, 1]), y_train, epochs=10, batch_size=32)
    print("모델 학습 완료")
    return model


def save_model(model, path):
    """
    모델을 파일에 저장

    Args:
        model (RecommenderModel): 저장할 모델
        path (str): 모델 저장 경로
    """
    print(f"모델을 {os.path.relpath(path)}에 저장합니다...")
    os.makedirs(MODEL_DIR, exist_ok=True)  # 디렉토리 생성
    model.save(path, include_optimizer=False)  # 옵티마이저 상태 제외
    model.save(path)
    print("모델 저장 완료")


def load_model(path):
    """
    저장된 모델을 로드
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {os.path.relpath(path)}")
        print(f"{os.path.relpath(path)}에서 모델을 로드합니다...")
        model = tf.keras.models.load_model(
            path, custom_objects={"RecommenderModel": RecommenderModel}, compile=False
        )
        model.compile(optimizer="adam", loss="mse")  # 옵티마이저 재설정
        print("모델 로드 완료")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    print(f"모델 저장 경로: {os.path.relpath(MODEL_PATH)}")

    # 모델 파일 존재 여부 확인
    if os.path.exists(MODEL_PATH):
        print(f"모델 파일이 존재합니다. 로드 시도 중: {os.path.relpath(MODEL_PATH)}")
        try:
            model = load_model(MODEL_PATH)
            print("모델 로드 성공!")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
    else:
        print("모델 파일이 없습니다. 학습을 시작합니다...")

        # 데이터베이스에서 데이터 로드
        print("데이터베이스에서 데이터 로드 중...")
        ai_user_data, location_data = load_data()
        max_user_id, max_location_id = get_max_ids()  # 최대 user_id, location_id 가져오기

        # 데이터 전처리
        data = ai_user_data[['user_id', 'location_id', 'rating']]
        data.columns = ['user_id', 'place_id', 'rating']

        print(f"데이터 전처리 완료: 사용자 ID 범위 [0, {max_user_id}], 장소 ID 범위 [0, {max_location_id}]")

        # 모델 학습
        model = train_model(data, max_user_id, max_location_id)

        # 모델 저장
        save_model(model, MODEL_PATH)
        print("모델 학습 및 저장 완료.")
