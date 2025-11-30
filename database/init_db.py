"""
데이터베이스 초기화 스크립트
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models import db

def init_database():
    """데이터베이스 테이블 생성"""
    app = create_app()
    
    with app.app_context():
        # 모든 테이블 생성
        db.create_all()
        print("✅ 데이터베이스 테이블이 성공적으로 생성되었습니다.")
        
        # 인덱스 생성 (MySQL의 경우)
        try:
            from sqlalchemy import text
            # created_at 인덱스
            db.session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_rubbings_created_at 
                ON rubbings(created_at DESC)
            """))
            # is_completed 인덱스
            db.session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_rubbings_is_completed 
                ON rubbings(is_completed)
            """))
            db.session.commit()
            print("✅ 인덱스가 성공적으로 생성되었습니다.")
        except Exception as e:
            # SQLite는 IF NOT EXISTS를 지원하지 않을 수 있음
            print(f"⚠️  인덱스 생성 중 오류 (무시 가능): {e}")

if __name__ == '__main__':
    try:
        init_database()
    except Exception as e:
        print(f"❌ 데이터베이스 초기화 실패: {e}")
        sys.exit(1)

