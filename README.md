## TripChoseAI : 여행지 추천 AI 모델

**TripChoseAI**는 사용자의 평점 데이터를 분석하여 개인화된 여행지 추천을 제공하는 Flask 기반 애플리케이션입니다.  
TensorFlow 기반의 추천 알고리즘을 사용하여 사용자 선호도를 반영하며, 간편한 JSON API로 통합 가능합니다.

---

![image](https://github.com/user-attachments/assets/a044c23d-82fa-4978-93b9-7b98dcc40a75)

![image](https://github.com/user-attachments/assets/fd80f542-2237-449a-aea8-d3392dad1eeb)

---

## 주요 특징

- **개인화된 추천**: 사용자의 평점 데이터를 분석하여 맞춤형 여행지 추천 제공  
- **심플한 API**: JSON 포맷의 응답으로 손쉽게 외부 시스템과 통합 가능  
- **확장 가능**: 새로운 데이터와 기능을 추가하기 용이한 설계  

---

## 주요 기술

| **기술**                 | **설명**                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| ![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white) |Programming Language                                   |
| ![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) |Backend                               |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) | Recommendation Engine                                        |
| ![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) | Data Handling                                            |
| ![Local Server](https://img.shields.io/badge/Local%20Server-%232D9CDB?style=for-the-badge) | Deployment                              |

---

## 프로젝트 구조

```plaintext
TripChose/
├── app/
│   ├── __init__.py            # Flask 앱 팩토리
│   ├── routes.py              # API 라우트 및 요청 처리
│   ├── models/
│   │   ├── trainer.py         # 추천 모델 학습, 저장, 로드
│   │   ├── recommender.py     # 추천 로직 구현
│   │   └── saved_model/       # 학습된 모델 저장 위치
├── data/
│   ├── users.csv              # 사용자 평점 데이터
│   ├── places.csv             # 여행지 정보 데이터
├── venv/                      # Python 가상 환경
├── requirements.txt           # 프로젝트 종속성
├── README.md                  # 프로젝트 설명서
└── run.py                     # Flask 서버 실행
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
- **`users.csv`**: 사용자-장소 평점 데이터
  ```csv
  user_id,place_id,rating
  1,1,5
  2,3,4
  ```
- **`places.csv`**: 여행지 정보 데이터
  ```csv
  id,name,category,location,rating
  1,Osaka Castle,Historical,Osaka,4.5
  2,Fushimi Inari Taisha,Nature,Kyoto,4.7
  ```

### 2. 추천 API
- **추천 요청**
  - `/recommend` 엔드포인트에 POST 요청으로 `user_id` 전달.
  ```json
  {
    "user_id": 1
  }
  ```
- **추천 응답**
  - 사용자 선호도를 기반으로 상위 3개의 여행지를 반환.
  ```json
  {
    "recommendations": [
        "Osaka Castle",
        "Fushimi Inari Taisha",
        "Kinkakuji"
    ]
  }
  ```

---

## 설치 및 실행

### 1. 환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 데이터 준비
`data/` 폴더에 `users.csv`와 `places.csv` 파일을 배치합니다.

### 3. 모델 학습
```bash
python app/models/trainer.py
```

### 4. 서버 실행
```bash
python run.py
```

### 5. API 테스트
```bash
curl -X POST http://127.0.0.1:5000/recommend -H "Content-Type: application/json" -d '{"user_id": 1}'
```

---

### **향후 목표**

1. **사용자 맞춤 추천 강화**
   - 협업 필터링 알고리즘, 하이브리드 추천 시스템을 추가하여 개인별 특성을 더 반영한 추천 알고리즘 구축 예정입니다.

2. **추천 모델의 속도 최적화**  
   - Pre-computation(사전 계산) 방식을 사용하여 알고리즘의 계산량을 줄일 계획입니다.
     
3. **TensorFlow Lite 적용**  
     - 추천 모델을 경량화하여 모바일 환경에서도 추천 시스템을 활용할 수 있도록 최적화할 계획입니다.

4. **데이터 정규화 및 확장**  
     - 평점 데이터를 0~1 범위로 정규화하여 모델의 학습 및 추론 정확도를 개선할 예정입니다.
