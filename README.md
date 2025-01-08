## TripChoseAI : 여행지 추천 AI 모델

**TripChoseAI**는 Flask 기반 백엔드와 TensorFlow를 활용한 AI 모델을 통해 사용자 맞춤 여행지를 추천하는 시스템입니다.
이 프로젝트는 여행 데이터를 바탕으로 사용자의 선호도를 분석하여 개인화된 추천 결과를 제공합니다.

---
### 실제 프로젝트 적용 화면
![Project](https://github.com/user-attachments/assets/e5ca53cc-4507-4a26-bec6-bdfd7a59e0cb)

---
### 모델 생성 화면
![RanderModel](https://github.com/user-attachments/assets/8c39a71c-ceb3-4112-9618-252de6fbde54)

---
### POSTMAN
![PostMan](https://github.com/user-attachments/assets/3cb21384-19b0-49e6-ae2c-8def2edd22b4)

---

## 주요 특징

- **개인화된 추천**: 사용자의 평점 데이터를 분석하여 맞춤형 여행지 추천 제공  
- **심플한 API**: JSON 포맷의 응답으로 손쉽게 외부 시스템과 통합 가능  
- **확장 가능**: 새로운 데이터와 기능을 추가하기 용이한 설계  

---

## 주요 기술

| **기술**                 | **설명**                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| ![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white) | Programming Language                                   |
| ![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) | Backend Framework                              |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) | Recommendation Engine                                        |
| ![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) | Data Handling                                            |
| ![MySQL](https://img.shields.io/badge/MySQL-%234479A1.svg?style=for-the-badge&logo=mysql&logoColor=white) | Database |
| ![Local Server](https://img.shields.io/badge/Local%20Server-%232D9CDB?style=for-the-badge) | Deployment Environment                              |

---

## 프로젝트 구조

```plaintext
TripChoseAI/
│
├── app/
│   ├── config/
│   │   ├── app_config.py   # 애플리케이션 설정 (모델 경로 등)
│   │   ├── db_config.py    # 데이터베이스 설정
│   ├── models/
│   │   ├── saved_model/    # 학습된 모델 저장 디렉토리
│   │   ├── trainer.py      # 추천 모델 학습 및 저장 로직
│   │   ├── recommender.py  # 추천 로직 구현
│   ├── utils/
│   │   ├── preprocess.py   # 데이터 전처리 
│   └── routes.py           # Flask 라우트 정의  
├── main.py                 # 애플리케이션 실행 스크립트
└── README.md               # 프로젝트 설명 파일

```

---

## 알고리즘

TripChoseAI는 **행렬 분해 기반 추천 알고리즘**을 사용합니다.

1. **사용자 및 장소 임베딩 학습**
   - TensorFlow의 `Embedding` 레이어를 사용하여 사용자와 장소를 고정 크기 벡터로 변환합니다.
   - 학습 과정에서 벡터는 사용자와 장소의 관계를 반영하도록 조정됩니다.

2. **유사도 계산**
   - 학습된 사용자와 장소 벡터 간의 내적(dot product)을 계산하여 선호도를 예측합니다.
   - 내적 결과는 높은 값을 가질수록 해당 장소에 대한 선호도가 높음을 의미합니다.

3. **평점 기반 정렬**
   - 예측된 평점을 기준으로 모든 장소를 내림차순 정렬합니다.
   - 상위 N개의 장소를 추천 목록으로 반환합니다.

---

## 주요 기능

### 1. 데이터 로드 및 전처리
- `preprocess.py`에서 제공되는 유틸리티를 활용하여 사용자와 장소 데이터를 병합 및 정제합니다.

### 2. 추천 API
- `/recommend` 엔드포인트를 통해 POST 요청으로 사용자 ID를 전달하면, 해당 사용자에 적합한 추천 결과를 반환합니다.

### 3. React, SpringBoot 프로젝트 연계

1. **랜덤 추천**
   - 회원가입시 React 프론트엔드에서 `/api/ai/random-places`를 호출하여 랜덤으로 추천된 여행지 5개를 가져옵니다.
   - 사용자가 각 장소에 대해 선호도를 입력합니다.
   - 입력된 평점은 `/api/ai/rating`을 통해 MySQL DB에 저장됩니다.

2. **추천 결과 확인**
   - 사용자가 메인 페이지에 접속하면 `/api/ai/verify` API가 호출됩니다.
   - React는 JWT 토큰을 스프링부트로 전송하여 이메일 기반 `userId`를 확인합니다.
   - 스프링부트는 `userId`를 바탕으로 Flask에 추천 요청을 보냅니다.
   - Flask는 학습된 모델로 추천된 여행지 4개를 반환하며, 이를 스프링부트는 DB의 여행지와 매칭시킨 후 데이터를 반환합니다.
   - React는 반환된 데이터를 표시합니다.

3. **예외 처리**
   - 비로그인 상태거나 모델 업데이트가 되지 않은 경우의 API 요청 시 Google 평점이 높은 여행지를 반환합니다.
   - Flask 서버 연결이 실패하면 로그를 기록하고 Google 평점 기반 데이터를 반환합니다.

4. **모델 업데이트**
   - Flask는 스케줄러를 통해 5분마다 DB에 저장된 학습 데이터를 사용해 모델을 업데이트합니다.

---


## 설치 및 실행 

### 1. 환경 설정
```bash
python -m venv venv
# Windows: venv\Scripts\activate
source venv/Scripts/activate 
pip install -r requirements.txt
```

### 2. 데이터 준비
- `db_config.py`에서 MySql 사용자 및 장소 데이터를 삽입합니다.
  - ##### 예시 데이터베이스 구조
      ##### `user` 테이블
      | **컬럼 이름**     | **데이터 타입** | **NULL 허용** | **키**      | **기타**             |
      |-------------------|-----------------|---------------|-------------|---------------------|
      | user_ai_id        | bigint          | NO            | PRI         | auto_increment      |
      | user_id           | bigint          | NO            | MUL         |                     |
      | location_id       | bigint          | NO            | MUL         |                     |
      | rating            | int             | NO            |             |                     |
    
      ##### `location` 테이블
      | **컬럼 이름**     | **데이터 타입** | **NULL 허용** | **키**      | **기타**             |
      |-------------------|-----------------|---------------|-------------|---------------------|
      | location_id       | bigint          | NO            | PRI         |                     |
      | location_name     | varchar         | NO            |             |                     |
      | google_rating     | float           | NO            |             |                     |

### 3. 모델 학습
```bash
python app/models/trainer.py
```

### 4. 서버 실행
```bash
python main.py
```

### 5. API 테스트
```bash
curl -X POST http://127.0.0.1:5000/recommend -H "Content-Type: application/json" -d '{"user_id": 1}'
```

---

## 향후 목표

- **추천 로직 개선**
  - 사용자 피드백을 활용한 추천 결과 개선.
  - 추가적인 컨텍스트(시간, 위치 등)를 고려한 추천.

- **확장성 있는 아키텍처 설계**
  - 대규모 데이터를 효율적으로 처리하기 위한 분산 시스템 도입.
  - API 응답 속도를 높이기 위해 캐싱 적용.

---

