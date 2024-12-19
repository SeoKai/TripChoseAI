import os
import pandas as pd
import tensorflow as tf

class RecommenderModel(tf.keras.Model):
    """
    추천 시스템 모델 정의.
    사용자, 장소의 임베딩 생성
    내적(dot)을 계산하여 선호도 예측

    Attributes:
        num_users (int): 사용자 수
        num_places (int): 장소 수
        embedding_dim (int): 임베딩 차원
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
        """
        모델의 forward 패스.
        사용자, 장소의 임베딩 계산
        내적 결과 반환

        Args:
            inputs (tuple or Tensor): 사용자 ID와 장소 ID

        Returns:
            Tensor: 예측된 선호도
        """
        if isinstance(inputs, tuple):
            user_ids, place_ids = inputs
        else:
            user_ids, place_ids = inputs[:, 0], inputs[:, 1]

        user_vector = self.user_embedding(user_ids)
        place_vector = self.place_embedding(place_ids)
        return self.dot([user_vector, place_vector])

    def get_config(self):
        """
        모델의 설정 정보를 반환

        Returns:
            dict: 모델 설정 정보
        """
        config = super(RecommenderModel, self).get_config()
        config.update({
            "num_users": self.num_users,
            "num_places": self.num_places,
            "embedding_dim": self.embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        설정 정보를 사용해 모델을 생성

        Args:
            config (dict): 모델 설정 정보

        Returns:
            RecommenderModel: 새로 생성된 모델
        """
        num_users = config.pop("num_users")
        num_places = config.pop("num_places")
        embedding_dim = config.pop("embedding_dim")
        return cls(num_users=num_users, num_places=num_places, embedding_dim=embedding_dim, **config)


def train_model(train_data, num_users, num_places):
    """
    추천 모델 학습

    Args:
        train_data (pd.DataFrame): 학습 데이터 (user_id, place_id, rating)
        num_users (int): 사용자 수
        num_places (int): 장소 수

    Returns:
        RecommenderModel: 학습된 모델
    """
    X_train = train_data[['user_id', 'place_id']].values
    y_train = train_data['rating'].values

    # 모델 초기화
    model = RecommenderModel(num_users, num_places)

    # 모델 구조 확인
    print("모델 구조:")
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    # 모델 학습
    print("모델 학습 시작")
    model.fit((X_train[:, 0], X_train[:, 1]), y_train, epochs=10, batch_size=32)
    print("모델 학습 완료")

    # 학습된 임베딩 가중치 출력
    print("사용자 임베딩 가중치:")
    print(model.user_embedding.get_weights()[0])

    print("장소 임베딩 가중치:")
    print(model.place_embedding.get_weights()[0])

    return model


def save_model(model, path):
    """
    모델을 파일에 저장

    Args:
        model (RecommenderModel): 저장할 모델
        path (str): 모델 저장 경로
    """
    print(f"모델을 {path}에 저장합니다...")
    model.save(path)
    print("모델 저장 완료")


def load_model(path):
    """
    저장된 모델을 로드

    Args:
        path (str): 모델 경로

    Returns:
        RecommenderModel: 로드된 모델
    """
    try:
        print(f"{path}에서 모델을 로드합니다...")
        model = tf.keras.models.load_model(
            path, custom_objects={"RecommenderModel": RecommenderModel}
        )
        print("모델 로드 완료")
        print(f"num_users: {model.num_users}, num_places: {model.num_places}")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    # 경로 설정
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    user_file = os.path.join(BASE_DIR, "data", "users.csv")
    place_file = os.path.join(BASE_DIR, "data", "places.csv")
    save_path = os.path.join(BASE_DIR, "app", "models", "saved_model", "recommender_model.keras")

    # 데이터 로드
    print("데이터 로드 중...")
    users = pd.read_csv(user_file)
    places = pd.read_csv(place_file)

    # 데이터 병합 및 전처리
    data = pd.merge(users, places, left_on="place_id", right_on="id")
    data = data.rename(columns={"rating_x": "rating"})
    data = data[['user_id', 'place_id', 'rating']]
    data['user_id'] -= 1  # 사용자 ID를 0부터 시작하도록 조정
    data['place_id'] -= 1  # 장소 ID를 0부터 시작하도록 조정

    print("전처리된 데이터 샘플:")
    print(data.head())

    # 사용자 및 장소 수 계산
    num_users = data['user_id'].nunique()
    num_places = data['place_id'].nunique()
    print(f"사용자 수: {num_users}, 장소 수: {num_places}")

    # 모델 학습
    model = train_model(data, num_users, num_places)

    # 모델 저장 경로
    save_path = os.path.join(BASE_DIR, "app", "models", "saved_model", "recommender_model.keras")

    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_model(model, save_path)

    # 저장 후 모델 출력
    model.summary()
