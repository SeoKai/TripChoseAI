import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(user_file, place_file):
    """
    사용자 및 여행지 데이터를 로드하여 병합한 후 학습 데이터와 테스트 데이터로 분리

    Args:
        user_file (str): 사용자 데이터가 저장된 CSV 파일 경로
        place_file (str): 여행지 데이터가 저장된 CSV 파일 경로

    Returns:
        tuple: train_data, test_data
            train_data (pd.DataFrame): 학습에 사용할 데이터
            test_data (pd.DataFrame): 테스트에 사용할 데이터
    """
    # 사용자 데이터 로드
    users = pd.read_csv(user_file)
    # 여행지 데이터 로드
    places = pd.read_csv(place_file)

    # 사용자 데이터와 여행지 데이터 병합
    data = pd.merge(users, places, left_on='place_id', right_on='id')

    # 필요한 열만 선택
    data = data[['user_id', 'place_id', 'rating']]

    # 학습 데이터와 테스트 데이터 분리
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    return train_data, test_data
