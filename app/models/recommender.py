import tensorflow as tf

def recommend_places(model, user_id, num_places, places):
    """
    특정 사용자 추천 여행지 예측 후 반환

    Args:
        model (RecommenderModel): 학습된 추천 모델
        user_id (int): 추천을 받을 사용자 ID
        num_places (int): 전체 장소 개수
        places (list): 장소 이름 리스트

    Returns:
        list: 추천된 장소 이름 리스트
    """
    # 입력 유효성 검사
    if user_id < 0 or user_id >= model.num_users:
        raise ValueError(f"user_id {user_id}는 모델의 사용자 범위를 초과합니다. "
                         f"(모델 사용자 수: {model.num_users})")
    if num_places <= 0 or not places:
        raise ValueError("유효한 장소 개수 또는 장소 리스트를 제공해야 합니다.")

    try:
        # 입력 데이터 생성
        user_array = tf.constant([user_id] * num_places)
        place_array = tf.constant(range(num_places))

        # 모델 예측 수행
        predictions = model.predict([user_array, place_array])

        # 예측 점수 기준 정렬
        sorted_indices = tf.argsort(predictions[:, 0], direction='DESCENDING')
        top_indices = sorted_indices[:4].numpy()

        # 상위 4개의 추천 장소 반환
        recommendations = [places[i] for i in top_indices]
        return recommendations

    except Exception as e:
        # 모든 예외를 한 곳에서 처리
        raise RuntimeError(f"추천 로직 실행 중 오류 발생: {e}")
