# Epitext Backend

탁본 복원 시스템의 백엔드 API 서버입니다.

## 기술 스택

- **Framework**: Flask 3.0.0
- **ORM**: SQLAlchemy
- **Database**: MySQL (또는 SQLite)
- **Image Processing**: Pillow

## 설치 및 실행

### 1. 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# 데이터베이스 설정
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=epitext_db

# Flask 설정
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# 파일 업로드 설정
UPLOAD_FOLDER=./uploads
MAX_CONTENT_LENGTH=16777216  # 16MB

# 이미지 저장 경로
IMAGES_FOLDER=./images/rubbings
CROPPED_IMAGES_FOLDER=./images/rubbings/cropped
```

**참고**: MySQL을 사용하지 않는 경우, `DB_PASSWORD`를 비워두면 SQLite가 자동으로 사용됩니다.

### 4. 데이터베이스 생성 및 마이그레이션

```bash
python database/init_db.py
python database/seed_data.py  # 체리피킹 데이터 추가 (선택사항)
```

### 5. 서버 실행

```bash
python app.py
```

서버는 기본적으로 `http://localhost:8000`에서 실행됩니다.

## API 엔드포인트

### 탁본 목록

- `GET /api/rubbings` - 탁본 목록 조회 (필터링 지원)

### 탁본 상세

- `GET /api/rubbings/:id` - 탁본 상세 정보 조회
- `GET /api/rubbings/:id/download` - 탁본 원본 파일 다운로드
- `GET /api/rubbings/:id/statistics` - 탁본 통계 조회
- `GET /api/rubbings/:id/inspection-status` - 검수 상태 조회

### 복원 대상

- `GET /api/rubbings/:id/restoration-targets` - 복원 대상 목록 조회
- `GET /api/rubbings/:id/targets/:targetId/candidates` - 후보 한자 목록 조회
- `GET /api/rubbings/:id/targets/:targetId/reasoning` - 유추 근거 데이터 조회
- `GET /api/rubbings/:id/targets/:targetId/cropped-image` - 크롭 이미지 조회

### 탁본 업로드 및 검수

- `POST /api/rubbings/upload` - 탁본 이미지 업로드
- `POST /api/rubbings/:id/targets/:targetId/inspect` - 검수 결과 저장
- `POST /api/rubbings/complete` - 복원 완료 처리

자세한 API 명세는 `../Epitext_Front/BACKEND_IMPLEMENTATION_GUIDE.md`를 참고하세요.

## 프로젝트 구조

```
Epitext_Back/
├── app.py                 # Flask 앱 진입점
├── config.py              # 설정 파일
├── models.py              # 데이터베이스 모델
├── routes/                # API 라우트
│   ├── __init__.py
│   ├── rubbings.py
│   ├── targets.py
│   └── inspection.py
├── utils/                 # 유틸리티 함수
│   ├── __init__.py
│   ├── status_calculator.py
│   └── image_processor.py
├── database/              # 데이터베이스 관련
│   ├── init_db.py
│   └── seed_data.py
├── uploads/               # 업로드된 파일 (gitignore)
├── images/                # 이미지 저장소 (gitignore)
└── requirements.txt       # Python 의존성
```

## 개발 가이드

백엔드 구현 가이드는 `../Epitext_Front/BACKEND_IMPLEMENTATION_GUIDE.md`에 상세히 정리되어 있습니다.
