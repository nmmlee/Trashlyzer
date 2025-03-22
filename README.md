# Trashlyzer - AI 기반 대형 폐기물 배출 가이드 서비스
![로고](https://github.com/nmmlee/Trashlyzer/blob/main/static/Image/Trashlyzer_Logo.png?raw=true)
## 프로젝트 인원

| 이름       | 담당 파트                |
|-----------|-----------------------|
| 김정민(팀장) | 백엔드, 아키텍쳐 설계, 프로젝트 아이디어 제공   |
| 황지환     | AI(llama.cpp 최적화,키워드 분석 LLM고안,유사도검색) |
| 하민용     | 웹 UI/UX 디자인(반응형)          |

## 프로젝트 개요
**Trashlyzer**는 AI를 활용하여 사용자가 버리려는 대형 폐기물의 배출 방법과 비용을 쉽게 안내하는 웹 서비스입니다. 
대형 폐기물 배출 기준이 지역마다 다르고 정보 접근이 어려운 문제를 해결하고자 개발되었습니다.

## 주요 기능
### AI 기반 대형 폐기물 분석
1. 텍스트 및 이미지 입력을 통해 폐기물 종류를 자동으로 인식
1. Gemini Flash 모델로 이미지 분석, Qwen API로 텍스트 키워드 추출, 캐싱 테이블 확인 후 hit시 데이터 출력, miss시 유사도 검색 후 Llama 3.1 답변 생성.

### 구청 배출 기준 자동 매칭
1. AI 분석 결과를 지역별 대형 폐기물 배출 기준과 비교하여 최적의 배출 방법 추천
1. CSV 데이터베이스를 활용한 실시간 배출 기준 매칭

### 실시간 크롤링 & 캐싱
1. 구청 공식 배출 기준을 자동 크롤링하여 최신화
1. 캐시 시스템을 통한 동일 요청 시 빠른 응답 제공

### 비용 절감 및 성능 최적화
1. 로컬 AI (Llama.cpp)를 활용한 텍스트 요약 처리
1. 중복 요청 방지를 위한 IP 기반 요청 제한
1. 최대 업로드 용량(10MB) 제한 및 이미지 압축 기능 적용

## 기술 스택
### 프론트엔드
- **HTML/CSS/JavaScript** - UI 개발(media query 적용)
- **showdown cdn** - markdown 문법 적용한 응답 변환

### 백엔드
- **Node.js (Express)** - API 서버
- **FastAPI (Python)** - AI 데이터 처리
- **Python 크롤러** - 구청 데이터 자동 수집

### AI 모델
- **Gemini Flash** - 이미지 분석
- **Qwen API** - 텍스트 분석
- **Llama.cpp** - 텍스트 요약 및 자연어 처리

## 프로젝트 구조
```
📂 Trashlyzer
├── 📂 static           # 정적 파일 디렉토리
│   ├── style.css
│   └── index.html
│
├── 📂 data            # 데이터 파일
│   ├── 대형폐기물분류표_노원.csv
│   ├── 대형폐기물분류표_정제.csv
│   └── cache_memory.csv
│
├── server.js          # Express 서버
├── llm_server.py      # AI 서버 메인
├── llm_server_data_processing.py  # 데이터 처리
├── llm_server_utilities.py        # 유틸리티 함수
├── crawler.py         # 크롤러
└── README.md
```

## API 명세
### 대형 폐기물 분석 API
#### 요청 (POST `/ask`)
```json
{
  "text": "유저 질의",
  "image": "유저가 올린 해당 대형쓰레기 이미지 파일"
}
```

#### 응답
```json
{
  "llm_response" : "유저가 올린 이미지, 텍스트를 바탕으로 나온 응답"
}
```

## 라이선스
이 프로젝트는 Meta의 Llama 3.1 모델과 Alibaba의 Qwen 2.5 모델을 사용합니다.

라이선스 정보:
- Llama 3.1: Meta의 Llama 3.1 License에 따라 제공됩니다
  https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
- Qwen 2.5: Apache 2.0 License에 따라 제공됩니다.
  https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/LICENSE

  
Copyright © Meta Platforms, Inc. / Alibaba Group Holding Limited.

- showdown cdn: MIT License에 따라 제공됩니다.
  https://github.com/showdownjs/showdown/blob/master/LICENSE
