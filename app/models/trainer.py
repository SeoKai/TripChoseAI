from app.config.app_config import MODEL_DIR, MODEL_PATH
import os
import tensorflow as tf
from app.config.db_config import load_data, get_max_ids


class RecommenderModel(tf.keras.Model):
    """
    딥러닝 기반 협업 필터링 추천 모델 정의 (행렬 분해 방식)

    사용자 ID와 장소 ID를 각각 임베딩 벡터로 변환한 뒤,
    두 벡터 간의 내적(Dot Product)을 통해 예상 평점을 계산하는 구조로,
    전통적인 Matrix Factorization을 신경망으로 구현한 형태입니다
    """
    def __init__(self, num_users, num_places, embedding_dim=50, **kwargs):

        """
        모델 초기화 함수

        Args:
            num_users (int): 사용자 수 (user_id는 0부터 시작)
            num_places (int): 장소 수 (place_id는 0부터 시작)
            embedding_dim (int): 임베딩 차원 (default: 50)
        """
        super(RecommenderModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_dim = embedding_dim

        # 사용자 ID를 임베딩 벡터로 변환하는 레이어
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)

        # 장소 ID를 임베딩 벡터로 변환하는 레이어
        self.place_embedding = tf.keras.layers.Embedding(num_places, embedding_dim)

        # 사용자 벡터와 장소 벡터 간 내적을 통해 평점을 예측하는 레이어
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, inputs):
        user_ids, place_ids = inputs
        user_vector = self.user_embedding(user_ids)
        place_vector = self.place_embedding(place_ids)
        return self.dot([user_vector, place_vector])

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_places": self.num_places,
            "embedding_dim": self.embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def train_model(train_data, max_user_id, max_location_id, epochs=10, batch_size=32):
    """
    추천 모델 학습
    """
    num_users = max_user_id + 1
    num_places = max_location_id + 1

    model = RecommenderModel(num_users, num_places)
    X_train = train_data[['user_id', 'place_id']].values
    y_train = train_data['rating'].values

    model.compile(optimizer='adam', loss='mse')
    print(f"모델 학습 시작 (에포크: {epochs}, 배치 크기: {batch_size})")
    model.fit((X_train[:, 0], X_train[:, 1]), y_train, epochs=epochs, batch_size=batch_size)
    print("모델 학습 완료")
    return model


def save_model(model, path):
    """
    모델을 파일에 저장
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"모델을 저장합니다: {path}")
    model.save(path, include_optimizer=False)
    print("모델 저장 완료")


def load_model(path):
    """
    저장된 모델 로드
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    print(f"모델을 로드합니다: {path}")
    model = tf.keras.models.load_model(path, custom_objects={"RecommenderModel": RecommenderModel}, compile=False)
    model.compile(optimizer="adam", loss="mse")
    print("모델 로드 완료")
    return model


if __name__ == "__main__":
    print(f"모델 경로 확인: {MODEL_PATH}")

    # 항상 학습을 다시 실행
    print("모델 학습을 시작합니다...")
    ai_user_data, location_data = load_data()
    max_user_id, max_location_id = get_max_ids()

    train_data = ai_user_data[['user_id', 'location_id', 'rating']]
    train_data.columns = ['user_id', 'place_id', 'rating']

    model = train_model(train_data, max_user_id, max_location_id)
    save_model(model, MODEL_PATH)
    print("모델 학습 및 저장 완료")
