# 백엔드 설정 및 실행 가이드

## 📋 전체 프로세스 개요

1. **백엔드 환경 설정** → 2. **데이터베이스 초기화** → 3. **백엔드 서버 실행** → 4. **프론트엔드 연동**

---

## 1단계: 백엔드 환경 설정

### 1-1. Epitext_Back 폴더로 이동

```bash
cd "/Users/jincerity/Desktop/고려대 부트캠프/Epitext_Back"
```

### 1-2. Python 가상 환경 생성

```bash
python3 -m venv venv
```

**설명**: 가상 환경을 만들어 프로젝트별로 독립적인 Python 패키지 환경을 구성합니다.

### 1-3. 가상 환경 활성화

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

**활성화 확인**: 터미널 프롬프트 앞에 `(venv)`가 표시되면 성공입니다.

### 1-4. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

**설명**: Flask, SQLAlchemy, Pillow 등 필요한 라이브러리를 설치합니다.

**예상 설치 시간**: 1-2분

---

## 2단계: 환경 변수 설정

### 2-1. .env 파일 생성

프로젝트 루트(`Epitext_Back`)에 `.env` 파일을 생성합니다.

```bash
# macOS/Linux
touch .env

# Windows
type nul > .env
```

또는 텍스트 에디터로 직접 생성하세요.

### 2-2. .env 파일 내용 작성

`.env` 파일에 다음 내용을 복사하여 붙여넣으세요:

```bash
# 데이터베이스 설정
# MySQL을 사용하지 않는 경우, DB_PASSWORD를 비워두면 SQLite가 자동으로 사용됩니다
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=epitext_db

# Flask 설정
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production

# 파일 업로드 설정
UPLOAD_FOLDER=./uploads
MAX_CONTENT_LENGTH=16777216

# 이미지 저장 경로
IMAGES_FOLDER=./images/rubbings
CROPPED_IMAGES_FOLDER=./images/rubbings/cropped
```

**중요**:

- MySQL을 사용하는 경우: `DB_PASSWORD`에 실제 비밀번호를 입력하세요.
- MySQL을 사용하지 않는 경우: `DB_PASSWORD`를 비워두면 SQLite가 자동으로 사용됩니다 (개발용으로 권장).

---

## 3단계: 데이터베이스 초기화

### 3-1. 데이터베이스 테이블 생성

```bash
python database/init_db.py
```

**예상 출력**:

```
✅ 데이터베이스 테이블이 성공적으로 생성되었습니다.
✅ 인덱스가 성공적으로 생성되었습니다.
```

**설명**:

- SQLite를 사용하는 경우: `epitext_db.db` 파일이 생성됩니다.
- MySQL을 사용하는 경우: 먼저 데이터베이스를 생성해야 합니다:
  ```sql
  CREATE DATABASE epitext_db;
  ```

### 3-2. 체리피킹 데이터 추가 (선택사항)

```bash
python database/seed_data.py
```

**예상 출력**:

```
✅ 5개의 체리피킹 데이터가 추가되었습니다.
```

**설명**: 테스트용 샘플 데이터 5개를 추가합니다. 나중에 실제 데이터로 대체할 수 있습니다.

---

## 4단계: 백엔드 서버 실행

### 4-1. 서버 시작

```bash
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

### 4-2. 서버 동작 확인

브라우저에서 다음 URL을 열어보세요:

- `http://localhost:8000/` → `{"message": "Epitext Backend API", "version": "1.0.0"}`
- `http://localhost:8000/health` → `{"status": "healthy"}`
- `http://localhost:8000/api/rubbings` → 탁본 목록 (JSON 배열)

**성공**: JSON 응답이 보이면 백엔드가 정상 작동 중입니다! ✅

---

## 5단계: 프론트엔드 연동

### 5-1. 프론트엔드 환경 변수 확인

`Epitext_Front` 폴더의 `.env` 파일(또는 `.env.local`)에 다음이 설정되어 있는지 확인:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

없다면 생성하세요.

### 5-2. 프론트엔드에서 실제 API 호출로 전환

현재 `src/api/requests.js`에서 mock 데이터를 사용 중입니다. 실제 API를 호출하도록 변경해야 합니다.

**변경 전** (현재):

```javascript
export const getRubbingList = async (status = null) => {
  // TODO: 백엔드 API 연결 시 주석 해제
  // try {
  //   const response = await apiClient.get("/api/rubbings", { params });
  //   return response.data;
  // } catch (error) {
  //   console.error("Failed to fetch rubbings:", error);
  //   throw error;
  // }

  // 더미 데이터로 테스트
  return new Promise((resolve) => {
    setTimeout(() => {
      // ... mock 데이터 반환
    }, 1000);
  });
};
```

**변경 후**:

```javascript
export const getRubbingList = async (status = null) => {
  try {
    const params = status ? { status } : {};
    const response = await apiClient.get("/api/rubbings", { params });

    // 백엔드 응답을 프론트엔드 형식으로 변환
    const formattedData = response.data.map((item) => ({
      id: item.id,
      status: item.status,
      date: formatDate(item.created_at),
      restorationStatus: item.restoration_status || "-",
      processingTime: formatProcessingTime(item.processing_time),
      damageLevel: item.damage_level ? `${item.damage_level}%` : "-",
      inspectionStatus: item.inspection_status || "-",
      reliability: item.average_reliability ? `${item.average_reliability}%` : "-",
      is_completed: item.is_completed,
      image_url: item.image_url,
      filename: item.filename,
    }));

    return formattedData;
  } catch (error) {
    console.error("Failed to fetch rubbings:", error);
    throw error;
  }
};
```

### 5-3. 프론트엔드 서버 실행

새 터미널 창을 열고:

```bash
cd "/Users/jincerity/Desktop/고려대 부트캠프/Epitext_Front"
npm run dev
```

### 5-4. 연동 테스트

1. 브라우저에서 프론트엔드 앱 열기 (보통 `http://localhost:5173`)
2. 개발자 도구(F12) → Network 탭 열기
3. 탁본 목록 페이지로 이동
4. `GET /api/rubbings` 요청이 보이고 응답이 정상인지 확인

---

## 🔧 문제 해결

### 문제 1: "ModuleNotFoundError: No module named 'flask'"

**원인**: 가상 환경이 활성화되지 않았거나 의존성이 설치되지 않음

**해결**:

```bash
source venv/bin/activate  # 가상 환경 활성화
pip install -r requirements.txt  # 의존성 재설치
```

### 문제 2: "OperationalError: no such table"

**원인**: 데이터베이스 테이블이 생성되지 않음

**해결**:

```bash
python database/init_db.py  # 테이블 재생성
```

### 문제 3: "CORS error" (프론트엔드에서)

**원인**: CORS 설정 문제

**해결**: `app.py`에서 CORS가 이미 설정되어 있습니다. 백엔드 서버가 실행 중인지 확인하세요.

### 문제 4: "Connection refused" (프론트엔드에서)

**원인**: 백엔드 서버가 실행되지 않음

**해결**:

1. 백엔드 서버가 `http://localhost:8000`에서 실행 중인지 확인
2. `VITE_API_BASE_URL` 환경 변수가 올바른지 확인

---

## 📝 다음 작업

백엔드와 프론트엔드가 연동되면:

1. **나머지 API 엔드포인트 연동**

   - 탁본 상세 정보 조회
   - 복원 대상 목록 조회
   - 후보 한자 목록 조회
   - 검수 결과 저장
   - 등등...

2. **AI 모델 통합**

   - OCR 모델
   - 구두점 복원 모델
   - Vision/NLP 모델
   - 이미지 크롭 로직

3. **실제 데이터 처리**
   - 탁본 이미지 업로드 처리
   - AI 모델 결과를 DB에 저장
   - 검수 결과 업데이트

---

## 💡 팁

- **두 개의 터미널 사용**: 하나는 백엔드(`python app.py`), 다른 하나는 프론트엔드(`npm run dev`)
- **개발자 도구 활용**: 브라우저 개발자 도구의 Network 탭에서 API 요청/응답 확인
- **로그 확인**: 백엔드 터미널에서 요청 로그 확인 가능
- **SQLite 사용 권장**: 개발 초기에는 MySQL 설정 없이 SQLite로 시작하는 것이 간단합니다
