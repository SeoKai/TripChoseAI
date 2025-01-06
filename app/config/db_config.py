import pandas as pd
from sqlalchemy import create_engine, text

# SQLAlchemy 엔진 설정
db_config = {
    "host": "localhost",        # MySQL 서버 호스트
    "user": "root",             # 사용자 이름
    "password": "1234",         # 비밀번호
    "database": "testdb"        # 데이터베이스 이름
}

# SQLAlchemy 연결 URI
connection_uri = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
engine = create_engine(connection_uri)

# 데이터베이스에서 tbl_ai_user와 필터링된 tbl_location 가져오기
def load_data():
    """
    데이터베이스에서 데이터를 로드합니다.
    """
    # tbl_ai_user 데이터 가져오기
    ai_user_query = "SELECT user_id, location_id, rating FROM tbl_ai_user"
    ai_user_data = pd.read_sql(ai_user_query, con=engine)

    # 음식 태그를 제외한 tbl_location 데이터 가져오기
    location_query = """
        SELECT 
            l.location_id AS id, 
            l.location_name AS name, 
            l.google_rating AS rating
        FROM 
            tbl_location l
        JOIN 
            tbl_location_tag lt ON l.location_id = lt.location_id
        JOIN 
            tbl_tag t ON lt.tag_id = t.tag_id
        WHERE 
            t.tag_name IN ('관광명소', '랜드마크', '문화', '쇼핑') -- 선택된 태그만 포함
            AND l.location_id NOT IN (
                SELECT lt.location_id 
                FROM tbl_location_tag lt
                JOIN tbl_tag t ON lt.tag_id = t.tag_id
                WHERE t.tag_name = '음식' -- 음식 태그가 포함된 장소 제외
            )
        GROUP BY 
            l.location_id, l.location_name, l.google_rating;
    """
    location_data = pd.read_sql(location_query, con=engine)

    return ai_user_data, location_data


def get_max_ids():
    """
    데이터베이스에서 최대 user_id와 location_id 가져오기
    """
    query = text("""
        SELECT 
            (SELECT MAX(user_id) FROM tbl_ai_user) AS max_user_id,
            (SELECT MAX(location_id) FROM tbl_location) AS max_location_id
    """)

    with engine.connect() as connection:
        result = connection.execute(query).fetchone()

    # 튜플 인덱스로 접근
    return result[0], result[1]

# 데이터 로드 및 확인
if __name__ == "__main__":
    ai_user_data, location_data = load_data()

    print("AI User Data:")
    print(ai_user_data.head())

    print("\nFiltered Location Data:")
    print(location_data.head(10))

    max_user_id, max_location_id = get_max_ids()
    print(f"\nMax User ID: {max_user_id}, Max Location ID: {max_location_id}")
