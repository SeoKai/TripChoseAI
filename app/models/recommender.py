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
        print(f"user_id {user_id}는 모델의 사용자 범위를 초과합니다. 모델 사용자 수: {model.num_users}")
        raise ValueError(f"user_id {user_id}는 모델의 사용자 범위를 초과합니다.")

    # 입력 데이터 생성: 동일한 user_id와 모든 place_id를 매칭
    try:
        user_array = tf.constant([user_id] * num_places)
        place_array = tf.constant(range(num_places))
        print(f"user_array: {user_array.numpy()[:5]}... (총 {len(user_array)}개)")
        print(f"place_array: {place_array.numpy()[:5]}... (총 {len(place_array)}개)")
    except Exception as e:
        print(f"입력 데이터 생성 중 오류 발생: {e}")
        raise

    # 모델 예측 수행
    try:
        predictions = model.predict([user_array, place_array])
        print(f"모델 예측 결과: {predictions[:5]}... (총 {len(predictions)}개)")
    except Exception as e:
        print(f"모델 예측 중 오류 발생: {e}")
        raise ValueError(f"예측 중 오류 발생: {e}")

    # 예측 점수를 기준으로 장소 정렬
    try:
        sorted_indices = predictions[:, 0].argsort()[::-1]  # 내림차순으로 정렬
        top_indices = sorted_indices[:4]  # 상위 4개 장소 선택
        print(f"정렬된 인덱스: {sorted_indices[:10]}")
        print(f"선택된 인덱스: {top_indices}")
        recommendations = [places[i] for i in top_indices]
        print(f"추천 결과: {recommendations}")
        return recommendations
    except Exception as e:
        print(f"정렬 또는 추천 결과 생성 중 오류 발생: {e}")
        raise
