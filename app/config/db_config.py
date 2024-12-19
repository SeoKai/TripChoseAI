import pandas as pd
import mysql.connector

# MySQL 연결 설정
db_config = {
    "host": "localhost",        # MySQL 서버 호스트
    "user": "root",             # 사용자 이름
    "password": "1234",     # 비밀번호
    "database": "testdb"  # 데이터베이스 이름
}

# 데이터베이스에서 tbl_ai_user와 필터링된 tbl_location 가져오기
def load_data():
    # DB 연결
    conn = mysql.connector.connect(**db_config)

    # tbl_ai_user 데이터 가져오기
    ai_user_query = "SELECT user_id, location_id, rating FROM tbl_ai_user"
    ai_user_data = pd.read_sql(ai_user_query, conn)

    # 관광명소 태그에 해당하는 tbl_location 데이터 가져오기
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
            t.tag_name = '관광명소';
    """
    location_data = pd.read_sql(location_query, conn)

    # 연결 종료
    conn.close()

    return ai_user_data, location_data

# 데이터 로드 및 확인
ai_user_data, location_data = load_data()

print("AI User Data:")
print(ai_user_data.head())

print("\nFiltered Location Data:")
print(location_data.head())
