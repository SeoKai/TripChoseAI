import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(user_file, place_file, test_size=0.2, random_state=42):
    """
    사용자 및 여행지 데이터를 로드하여 병합한 후 학습 데이터와 테스트 데이터로 분리

    Args:
        user_file (str): 사용자 데이터가 저장된 CSV 파일 경로
        place_file (str): 여행지 데이터가 저장된 CSV 파일 경로
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 (기본값: 42)

    Returns:
        tuple: train_data, test_data
            train_data (pd.DataFrame): 학습에 사용할 데이터
            test_data (pd.DataFrame): 테스트에 사용할 데이터
    """
    try:
        # 경로 유효성 검사
        if not os.path.exists(user_file):
            raise FileNotFoundError(f"사용자 데이터 파일을 찾을 수 없습니다: {user_file}")
        if not os.path.exists(place_file):
            raise FileNotFoundError(f"여행지 데이터 파일을 찾을 수 없습니다: {place_file}")

        # 사용자 데이터 로드
        print(f"사용자 데이터 로드 중: {user_file}")
        users = pd.read_csv(user_file)

        # 여행지 데이터 로드
        print(f"여행지 데이터 로드 중: {place_file}")
        places = pd.read_csv(place_file)

        # 사용자 데이터와 여행지 데이터 병합
        print("데이터 병합 중...")
        data = pd.merge(users, places, left_on='place_id', right_on='id')
        print(f"병합된 데이터 크기: {data.shape}")

        # 필요한 열만 선택
        data = data[['user_id', 'place_id', 'rating']]

        # 학습 데이터와 테스트 데이터 분리
        print(f"데이터 분리: 테스트 비율 {test_size}, 랜덤 시드 {random_state}")
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

        print(f"학습 데이터 크기: {train_data.shape}, 테스트 데이터 크기: {test_data.shape}")
        return train_data, test_data

    except Exception as e:
        print(f"데이터 로드 또는 처리 중 오류 발생: {e}")
        raise
