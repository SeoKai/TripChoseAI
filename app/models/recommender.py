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
    # user_id가 모델 범위를 벗어나면 예외 발생
    if user_id < 0 or user_id >= model.num_users:
        raise ValueError(f"user_id {user_id}는 모델의 사용자 범위를 초과합니다.")

    # 입력 데이터 생성: 동일한 user_id와 모든 place_id를 매칭
    user_array = tf.constant([user_id] * num_places)
    place_array = tf.constant(range(num_places))

    # 모델 예측 수행
    try:
        predictions = model.predict([user_array, place_array])
    except Exception as e:
        raise ValueError(f"예측 중 오류 발생: {e}")

    # 예측 점수를 기준으로 장소 정렬
    sorted_indices = predictions[:, 0].argsort()[::-1]  # 내림차순으로 정렬
    top_indices = sorted_indices[:3]  # 상위 3개 장소 선택
    recommendations = [places[i] for i in top_indices]  # 장소 이름 매칭

    return recommendations
