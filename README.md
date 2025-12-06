# EPITEXT Backend

탁본 복원 시스템의 백엔드 API 서버입니다. AI 기반 OCR, 구두점 복원, 한자 복원, 번역 기능을 제공합니다.

## 📋 목차

- [시스템 요구사항](#시스템-요구사항)
- [설치 가이드](#설치-가이드)
- [모델 파일 준비](#모델-파일-준비)
- [환경 변수 설정](#환경-변수-설정)
- [데이터베이스 설정](#데이터베이스-설정)
- [서버 실행](#서버-실행)
- [API 엔드포인트](#api-엔드포인트)
- [문제 해결](#문제-해결)

## 🖥️ 시스템 요구사항

### 필수 요구사항

- **Python**: 3.8 이상 (3.9 이상 권장)
- **운영체제**: macOS, Linux, Windows (WSL 권장)
- **메모리**: 최소 8GB RAM (16GB 권장, AI 모델 로딩용)
- **디스크 공간**: 최소 10GB (모델 파일 포함)

### 권장 사양

- **GPU**: NVIDIA GPU with CUDA 지원 (선택사항, CPU도 가능하지만 느림)
- **Python 버전**: 3.9 ~ 3.11

## 📦 설치 가이드

### 1단계: 저장소 클론

```bash
git clone <repository-url>
cd Epitext_Back
```

### 2단계: Python 가상 환경 생성

```bash
# Python 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**활성화 확인**: 터미널 프롬프트 앞에 `(venv)`가 표시되면 성공입니다.

### 3단계: 의존성 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**예상 설치 시간**: 5-10분 (인터넷 속도에 따라 다름)

**주의사항**:
- PyTorch는 CPU 버전이 기본으로 설치됩니다. GPU를 사용하려면 [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)에서 CUDA 버전을 별도로 설치하세요.
- 설치 중 오류가 발생하면 Python 버전을 확인하세요 (3.8 이상 필요).

## 🤖 모델 파일 준비

EPITEXT는 다음 AI 모델들을 사용합니다. 각 모델 파일을 준비해야 합니다.

### 1. OCR 모델 파일 (필수)

OCR 모델은 다음 두 가지 파일이 필요합니다:

1. **Detection Model** (`best.pth`): 텍스트 영역 탐지 모델
2. **Recognition Model** (`best_5000.pt`): 한자 인식 모델

**파일 구조 예시**:
```
/path/to/ocr_weights/
├── best.pth                    # Detection 모델
├── best_5000.pt                # Recognition 모델
└── your-google-credentials.json # Google Vision API 인증 파일
```

**설정 방법**:
- `OCR_WEIGHTS_BASE_PATH` 환경 변수에 OCR 모델 파일들이 있는 폴더 경로를 설정합니다.
- 예: `OCR_WEIGHTS_BASE_PATH=/Users/username/ocr_weights`

### 2. Swin Transformer 체크포인트 (필수)

Swin Transformer 모델은 MASK2 복원에 사용됩니다.

**필요한 파일**:
- `best_model.pth` (우선 사용) 또는 `last_checkpoint.pth`

**파일 구조 예시**:
```
/path/to/swin_checkpoint/
├── best_model.pth              # 우선 사용 (있으면 이것 사용)
└── last_checkpoint.pth         # best_model.pth가 없으면 이것 사용
```

**설정 방법**:
- `SWIN_CHECKPOINT_PATH` 환경 변수에 체크포인트 파일이 있는 폴더 경로를 설정합니다.
- 예: `SWIN_CHECKPOINT_PATH=/Users/username/swin_checkpoint`
- `SWIN_MODEL_FILE` 환경 변수로 사용할 파일명을 지정할 수 있습니다 (기본값: `last_checkpoint.pth`)

### 3. Google Vision API 인증 파일 (필수)

Google Vision API를 사용하기 위해 서비스 계정 JSON 파일이 필요합니다.

**준비 방법**:
1. [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
2. Cloud Vision API 활성화
3. 서비스 계정 생성 및 JSON 키 다운로드
4. 다운로드한 JSON 파일을 `OCR_WEIGHTS_BASE_PATH` 폴더에 저장

**파일명**: `your-google-credentials.json` (원하는 이름으로 변경 가능)

### 4. Gemini API 키 (필수)

번역 기능을 위해 Google Gemini API 키가 필요합니다.

**준비 방법**:
1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 생성
2. 생성한 API 키를 `.env` 파일의 `GEMINI_TRANSLATION_API_KEY`에 설정

### 5. NLP 모델 (자동 다운로드)

NLP 모델은 Hugging Face에서 자동으로 다운로드됩니다:
- 구두점 복원: `seyoungsong/SikuRoBERTa-PUNC-AJD-KLC`
- MLM 모델: `jhangyejin/epitext-sikuroberta`

**인터넷 연결 필요**: 첫 실행 시 모델이 자동으로 다운로드됩니다.

## ⚙️ 환경 변수 설정

### 1. .env 파일 생성

프로젝트 루트(`Epitext_Back`)에 `.env` 파일을 생성합니다.

```bash
# macOS/Linux
touch .env

# Windows
type nul > .env
```

### 2. .env 파일 내용 작성

`.env.example` 파일을 참고하여 `.env` 파일을 작성합니다:

```bash
# ==============================================================================
# 데이터베이스 설정
# ==============================================================================
# MySQL을 사용하지 않는 경우, DB_PASSWORD를 비워두면 SQLite가 자동으로 사용됩니다
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=epitext_db

# ==============================================================================
# Flask 설정
# ==============================================================================
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here-change-in-production

# ==============================================================================
# 파일 업로드 설정
# ==============================================================================
UPLOAD_FOLDER=./uploads
MAX_CONTENT_LENGTH=16777216  # 16MB

# ==============================================================================
# 이미지 저장 경로
# ==============================================================================
IMAGES_FOLDER=./images/rubbings
CROPPED_IMAGES_FOLDER=./images/rubbings/cropped

# ==============================================================================
# OCR 모델 설정 (필수)
# ==============================================================================
# OCR 모델 파일들이 있는 기본 경로 (절대 경로 권장)
OCR_WEIGHTS_BASE_PATH=/path/to/ocr_weights

# Detection 모델 파일명 (OCR_WEIGHTS_BASE_PATH 아래에 있어야 함)
OCR_DETECTION_MODEL=best.pth

# Recognition 모델 파일명 (OCR_WEIGHTS_BASE_PATH 아래에 있어야 함)
OCR_RECOGNITION_MODEL=best_5000.pt

# Google Vision API 인증 파일명 (OCR_WEIGHTS_BASE_PATH 아래에 있어야 함)
GOOGLE_CREDENTIALS_JSON=your-google-credentials.json

# ==============================================================================
# Swin Transformer 모델 설정 (필수)
# ==============================================================================
# Swin 체크포인트 파일이 있는 폴더 경로 (절대 경로 권장)
SWIN_CHECKPOINT_PATH=/path/to/swin_checkpoint

# 사용할 모델 파일명 (best_model.pth 또는 last_checkpoint.pth)
# best_model.pth가 있으면 자동으로 우선 사용됨
SWIN_MODEL_FILE=last_checkpoint.pth

# ==============================================================================
# Gemini API 설정 (필수)
# ==============================================================================
# Google Gemini API 키 (번역 기능용)
GEMINI_TRANSLATION_API_KEY=your-gemini-api-key-here
```

### 3. 환경 변수 설정 예시

**macOS/Linux 예시**:
```bash
OCR_WEIGHTS_BASE_PATH=/Users/username/ocr_weights
SWIN_CHECKPOINT_PATH=/Users/username/swin_checkpoint
GEMINI_TRANSLATION_API_KEY=AIzaSy...
```

**Windows 예시**:
```bash
OCR_WEIGHTS_BASE_PATH=C:\Users\username\ocr_weights
SWIN_CHECKPOINT_PATH=C:\Users\username\swin_checkpoint
GEMINI_TRANSLATION_API_KEY=AIzaSy...
```

**중요 사항**:
- 경로는 **절대 경로**를 사용하는 것을 권장합니다.
- 상대 경로도 가능하지만, 프로젝트 루트 기준입니다.
- 경로에 공백이 있으면 따옴표로 감싸지 마세요 (환경 변수는 따옴표 없이 설정).

## 🗄️ 데이터베이스 설정

### SQLite 사용 (권장, 개발용)

SQLite는 별도 설치 없이 사용할 수 있습니다. `.env` 파일에서 `DB_PASSWORD`를 비워두면 자동으로 SQLite가 사용됩니다.

```bash
# .env 파일
DB_PASSWORD=
```

### MySQL 사용 (선택사항, 프로덕션용)

MySQL을 사용하려면:

1. MySQL 설치 및 실행
2. 데이터베이스 생성:
   ```sql
   CREATE DATABASE epitext_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```
3. `.env` 파일에 MySQL 정보 설정:
   ```bash
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_mysql_password
   DB_NAME=epitext_db
   ```

### 데이터베이스 초기화

```bash
# 가상 환경이 활성화되어 있는지 확인
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate    # Windows

# 데이터베이스 테이블 생성
python database/init_db.py
```

**예상 출력**:
```
✅ 데이터베이스 테이블이 성공적으로 생성되었습니다.
✅ 인덱스가 성공적으로 생성되었습니다.
```

**SQLite 사용 시**: `instance/epitext_db.db` 파일이 생성됩니다.

**MySQL 사용 시**: 위에서 생성한 데이터베이스에 테이블이 생성됩니다.

## 🚀 서버 실행

### 1. 서버 시작

```bash
# 가상 환경 활성화 확인
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate    # Windows

# 서버 실행
python app.py
```

**예상 출력**:
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:8000
Press CTRL+C to quit
```

### 2. 서버 동작 확인

브라우저 또는 터미널에서 다음 URL을 확인하세요:

```bash
# Health Check
curl http://localhost:8000/health

# API 루트
curl http://localhost:8000/
```

**예상 응답**:
- `/health`: `{"status": "healthy"}`
- `/`: `{"message": "Epitext Backend API", "version": "1.0.0"}`

### 3. 프론트엔드 연동

프론트엔드가 `http://localhost:5173`에서 실행 중이면, 백엔드는 `http://localhost:8000`에서 실행되어야 합니다.

CORS는 이미 설정되어 있으므로 별도 설정이 필요 없습니다.

## 📡 API 엔드포인트

### 탁본 목록

- `GET /api/rubbings` - 탁본 목록 조회 (필터링 지원)
  - Query Parameters: `status` (선택사항)

### 탁본 상세

- `GET /api/rubbings/:id` - 탁본 상세 정보 조회
- `GET /api/rubbings/:id/statistics` - 탁본 통계 조회
- `GET /api/rubbings/:id/inspection-status` - 검수 상태 조회

### 복원 대상

- `GET /api/rubbings/:id/restoration-targets` - 복원 대상 목록 조회
- `GET /api/rubbings/:id/targets/:targetId/candidates` - 후보 한자 목록 조회
- `GET /api/rubbings/:id/targets/:targetId/reasoning` - 유추 근거 데이터 조회

### 탁본 업로드 및 검수

- `POST /api/rubbings/upload` - 탁본 이미지 업로드
- `POST /api/rubbings/:id/targets/:targetId/inspect` - 검수 결과 저장

### 번역

- `GET /api/rubbings/:id/targets/:targetId/translation` - 번역 조회
- `POST /api/rubbings/:id/targets/:targetId/preview-translation` - 번역 미리보기

자세한 API 명세는 각 엔드포인트의 소스 코드를 참고하세요.

## 🔧 문제 해결

### 문제 1: "ModuleNotFoundError: No module named 'flask'"

**원인**: 가상 환경이 활성화되지 않았거나 의존성이 설치되지 않음

**해결**:
```bash
# 가상 환경 활성화
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows

# 의존성 재설치
pip install -r requirements.txt
```

### 문제 2: "OCR_WEIGHTS_BASE_PATH environment variable is required"

**원인**: `.env` 파일에 `OCR_WEIGHTS_BASE_PATH`가 설정되지 않음

**해결**:
1. `.env` 파일이 프로젝트 루트에 있는지 확인
2. `.env` 파일에 `OCR_WEIGHTS_BASE_PATH=/path/to/ocr_weights` 추가
3. 경로가 올바른지 확인 (절대 경로 권장)

### 문제 3: "SWIN_CHECKPOINT_PATH가 설정되지 않았습니다"

**원인**: `.env` 파일에 `SWIN_CHECKPOINT_PATH`가 설정되지 않음

**해결**:
1. `.env` 파일에 `SWIN_CHECKPOINT_PATH=/path/to/swin_checkpoint` 추가
2. 해당 경로에 `best_model.pth` 또는 `last_checkpoint.pth` 파일이 있는지 확인

### 문제 4: "GOOGLE_CREDENTIALS_JSON environment variable is required"

**원인**: Google Vision API 인증 파일 경로가 잘못되었거나 파일이 없음

**해결**:
1. `.env` 파일의 `GOOGLE_CREDENTIALS_JSON` 값이 올바른지 확인
2. 파일이 `OCR_WEIGHTS_BASE_PATH` 아래에 있는지 확인
3. 파일 경로: `{OCR_WEIGHTS_BASE_PATH}/{GOOGLE_CREDENTIALS_JSON}`

### 문제 5: "OperationalError: no such table"

**원인**: 데이터베이스 테이블이 생성되지 않음

**해결**:
```bash
python database/init_db.py
```

### 문제 6: 모델 로딩이 매우 느림

**원인**: CPU만 사용 중이거나 메모리 부족

**해결**:
- GPU가 있다면 PyTorch CUDA 버전 설치
- 메모리가 부족하면 다른 프로그램 종료
- 첫 실행 시 Hugging Face 모델 다운로드로 인한 지연일 수 있음 (정상)

### 문제 7: "CORS error" (프론트엔드에서)

**원인**: 백엔드 서버가 실행되지 않았거나 CORS 설정 문제

**해결**:
1. 백엔드 서버가 `http://localhost:8000`에서 실행 중인지 확인
2. 프론트엔드의 `VITE_API_BASE_URL`이 올바른지 확인

### 문제 8: 모델 파일을 찾을 수 없음

**원인**: 경로가 잘못되었거나 파일명이 다름

**해결**:
1. 환경 변수의 경로가 절대 경로인지 확인
2. 파일이 실제로 존재하는지 확인:
   ```bash
   ls /path/to/ocr_weights/best.pth
   ls /path/to/swin_checkpoint/best_model.pth
   ```
3. 파일명이 정확한지 확인 (대소문자 구분)

## 📁 프로젝트 구조

```
Epitext_Back/
├── app.py                      # Flask 앱 진입점
├── config.py                   # 설정 파일
├── models.py                   # 데이터베이스 모델
├── .env                        # 환경 변수 (gitignore)
├── .env.example                # 환경 변수 예시
├── requirements.txt             # Python 의존성
├── routes/                     # API 라우트
│   ├── __init__.py
│   ├── rubbings.py             # 탁본 관련 API
│   ├── targets.py              # 복원 대상 관련 API
│   ├── inspection.py           # 검수 관련 API
│   └── translation.py          # 번역 관련 API
├── ai_modules/                 # AI 모듈
│   ├── ocr_engine.py           # OCR 엔진
│   ├── nlp_engine.py           # NLP 엔진 (구두점, MLM)
│   ├── swin_engine.py          # Swin Transformer 엔진
│   ├── translation_engine.py   # 번역 엔진
│   ├── config/                 # 모델 설정 파일
│   └── models/                 # 모델 정의
├── database/                   # 데이터베이스 관련
│   ├── init_db.py              # DB 초기화
│   └── seed_data.py            # 샘플 데이터 (선택사항)
├── utils/                      # 유틸리티 함수
│   ├── status_calculator.py    # 상태 계산
│   └── image_processor.py      # 이미지 처리
├── uploads/                    # 업로드된 파일 (gitignore)
├── images/                     # 이미지 저장소 (gitignore)
│   └── rubbings/
│       ├── original/           # 원본 이미지
│       ├── processed/          # 처리된 이미지
│       └── cropped/            # 크롭된 이미지
└── instance/                   # SQLite DB (gitignore)
    └── epitext_db.db
```

## 📝 추가 정보

### 개발 모드 vs 프로덕션 모드

현재는 개발 모드로 설정되어 있습니다. 프로덕션 배포 시:

1. `FLASK_DEBUG=False`로 변경
2. `SECRET_KEY`를 강력한 랜덤 문자열로 변경
3. WSGI 서버 사용 (예: Gunicorn, uWSGI)
4. 환경 변수 보안 강화

### 로그 확인

서버 실행 시 터미널에 로그가 출력됩니다. AI 모델 로딩, 처리 과정 등을 확인할 수 있습니다.

### 성능 최적화

- GPU 사용 시 처리 속도가 크게 향상됩니다
- 첫 실행 시 Hugging Face 모델 다운로드로 시간이 걸릴 수 있습니다
- 모델은 싱글톤 패턴으로 한 번만 로드되므로, 서버 재시작 없이 여러 요청 처리 가능

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. 환경 변수가 올바르게 설정되었는지
2. 모델 파일이 올바른 경로에 있는지
3. 데이터베이스가 초기화되었는지
4. 모든 의존성이 설치되었는지

---

**EPITEXT Backend v1.0.0**
