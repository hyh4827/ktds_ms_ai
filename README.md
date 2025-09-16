# 📋 RFP 분석 시스템

Azure AI Search와 OpenAI를 활용한 지능형 RFP(제안요청서) 분석 도구입니다.

## 🎯 개발 목적

- **RFP 문서 자동 분석**: PDF, DOCX, TXT 형식의 RFP 문서를 자동으로 분석하여 구조화된 정보를 추출
- **지능형 질의응답**: RFP 내용에 대한 자연어 질문에 정확한 답변 제공
- **유사 RFP 검색**: 벡터 검색을 통한 유사한 RFP 문서 검색 및 참고

## ✨ 주요 기능

### 1. 📄 RFP 파일 분석
- **다양한 파일 형식 지원**: PDF, DOCX, TXT 파일 업로드 및 텍스트 추출
- **11개 카테고리별 구조화된 분석**:
  - 핵심 개요 (배경목적, 범위, 기대성과, 용어정의, 이해관계자)
  - 일정·마일스톤 (사업기간, 주요마일스톤, 제출물일정, 질의응답마감)
  - 예산·가격 (추정예산, 부가세포함, 가격구성, 지불조건, 원가산출근거)
  - 평가·선정 기준 (정량정성배점, 가점감점요건, 탈락필수요건)
  - 요구사항 (상세목록, 기능요구, 인터페이스연계, 데이터, 비기능요구, 호환성표준)
  - 보안·준법 (인증권한감사, 개인정보컴플라이언스, 망구성암호화, 취약점진단)
  - 서비스 수준·운영 (SLA, 장애변경배포, 모니터링리포팅, 헬프데스크, 교육매뉴얼)
  - 품질/검수·인수 기준 (산출물목록, 테스트계획, 인수기준, 파일럿PoC)
  - 계약·법무 (계약유형, 지적재산권, 비밀유지, 손해배상, 하자보수)
  - 공급사 자격·역량 (참여제한, 필수자격, 투입인력, 레퍼런스)
  - 제출 형식·지시 (제안서형식, 필수첨부, 제출채널, 프레젠테이션)

### 2. ❓ 질의응답
- **자연어 질문 처리**: RFP 내용에 대한 자유로운 질문
- **구조화된 답변**: 분석된 11개 카테고리 정보를 활용한 정확한 답변
- **예시 질문 제공**: 자주 묻는 질문들을 버튼으로 제공

### 3. 🔍 유사 RFP 검색
- **벡터 검색**: Azure AI Search의 벡터 검색 기능 활용
- **의미적 유사도**: 키워드 기반이 아닌 의미적 유사도로 검색
- **상세 정보 제공**: 유사한 RFP의 상세 정보 및 요구사항 표시

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **AI Services**: 
  - Azure OpenAI (GPT-4o-mini)
  - Azure AI Search
- **Document Processing**: 
  - PyPDF2 (PDF 처리)
  - python-docx (DOCX 처리)
- **Vector Search**: Azure AI Search Vector Search
- **Environment**: python-dotenv

## 📋 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd ktds_ai

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 정보를 입력하세요:

```env
# Azure AI Search 설정
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=rfp-index

# Azure OpenAI 설정
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### 3. 애플리케이션 실행

```bash
# Streamlit 웹 애플리케이션 실행
streamlit run app.py

# 또는 콘솔 애플리케이션 실행
python main.py
```

## 🚀 사용법

### 웹 애플리케이션 (Streamlit)

1. **서비스 초기화**
   - 사이드바에서 Azure AI Search와 OpenAI 설정 입력
   - "서비스 초기화" 버튼 클릭

2. **RFP 분석**
   - "RFP 파일 분석" 탭에서 PDF/DOCX/TXT 파일 업로드
   - "RFP 분석 시작" 버튼 클릭
   - 11개 카테고리별 분석 결과 확인

3. **질의응답**
   - "질의응답" 탭에서 RFP 내용에 대한 질문 입력
   - 예시 질문 버튼 활용 또는 직접 질문 입력
   - "답변 생성" 버튼 클릭

4. **유사 RFP 검색**
   - "유사 RFP 검색" 탭에서 "유사 RFP 검색" 버튼 클릭
   - 분석된 키워드를 기반으로 유사한 RFP 검색

### 콘솔 애플리케이션

```bash
python main.py
```

1. Azure 서비스 설정 정보 입력
2. 메뉴에서 원하는 기능 선택:
   - RFP 파일 분석
   - 유사 RFP 검색
   - 제안서 생성

## 📁 프로젝트 구조

```
ktds_ai/
├── app.py                 # Streamlit 웹 애플리케이션
├── main.py               # 콘솔 애플리케이션
├── rfp_analyzer.py       # RFP 분석 핵심 로직
├── sample.py             # 샘플 코드
├── requirements.txt      # Python 의존성
├── streamlit.sh         # Streamlit 실행 스크립트
├── sample.txt           # 샘플 텍스트 파일
└── README.md            # 프로젝트 문서
```

## 🔧 주요 클래스 및 함수

### RFPAnalyzer 클래스

- `initialize_services()`: Azure 서비스 초기화
- `extract_text_from_file()`: 파일에서 텍스트 추출
- `analyze_rfp_with_gpt()`: GPT를 사용한 RFP 분석
- `store_rfp_in_search()`: 분석 결과를 Azure AI Search에 저장
- `search_similar_rfps()`: 유사 RFP 검색
- `ask_question_about_rfp()`: RFP 질의응답

## 📊 지원 파일 형식

- **PDF**: PyPDF2를 사용한 텍스트 추출
- **DOCX**: python-docx를 사용한 텍스트 추출
- **TXT**: UTF-8 인코딩 텍스트 파일

## ⚠️ 주의사항

- Azure AI Search와 OpenAI 서비스가 필요합니다
- 대용량 PDF 파일의 경우 처리 시간이 오래 걸릴 수 있습니다
- API 사용량에 따라 비용이 발생할 수 있습니다

