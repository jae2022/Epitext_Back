"""
체리피킹 데이터 시드 스크립트
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models import db, Rubbing, RubbingDetail, RubbingStatistics, RestorationTarget, Candidate, InspectionRecord
from datetime import datetime
import json

def seed_data():
    """체리피킹 데이터 추가"""
    app = create_app()
    
    with app.app_context():
        # 기존 데이터 확인
        existing_count = Rubbing.query.count()
        if existing_count > 0:
            print(f"⚠️  이미 {existing_count}개의 탁본 데이터가 존재합니다. 시드 데이터를 추가하지 않습니다.")
            return
        
        # 시드 데이터 생성
        seed_rubbings = [
            {
                'id': 1,
                'image_url': '/images/rubbings/rubbing_1.jpg',
                'filename': '귀법사적수화현응모지명.jpg',
                'created_at': datetime(2025, 10, 28, 6, 30, 0),
                'status': '우수',
                'restoration_status': '356자 / 복원 대상 23자',
                'processing_time': 222,
                'damage_level': 6.5,
                'inspection_status': '12자 완료',
                'average_reliability': 92.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 6, 33, 42)
            },
            {
                'id': 2,
                'image_url': '/images/rubbings/rubbing_2.jpg',
                'filename': '귀법사적수화현응모지명_2.jpg',
                'created_at': datetime(2025, 10, 28, 7, 0, 0),
                'status': '양호',
                'restoration_status': '68자 / 복원 대상 12자',
                'processing_time': 201,
                'damage_level': 17.6,
                'inspection_status': '12자 완료',
                'average_reliability': 76.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 7, 3, 21)
            },
            {
                'id': 3,
                'image_url': '/images/rubbings/rubbing_3.jpg',
                'filename': '귀법사적수화현응모지명_3.jpg',
                'created_at': datetime(2025, 10, 28, 7, 30, 0),
                'status': '우수',
                'restoration_status': '112자 / 복원 대상 8자',
                'processing_time': 225,
                'damage_level': 7.1,
                'inspection_status': '5자 완료',
                'average_reliability': 92.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 7, 33, 45)
            },
            {
                'id': 4,
                'image_url': '/images/rubbings/rubbing_4.jpg',
                'filename': '귀법사적수화현응모지명_4.jpg',
                'created_at': datetime(2025, 10, 28, 8, 0, 0),
                'status': '미흡',
                'restoration_status': '89자 / 복원 대상 31자',
                'processing_time': 302,
                'damage_level': 34.8,
                'inspection_status': '31자 완료',
                'average_reliability': 68.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 8, 5, 2)
            },
            {
                'id': 5,
                'image_url': '/images/rubbings/rubbing_5.jpg',
                'filename': '귀법사적수화현응모지명_5.jpg',
                'created_at': datetime(2025, 10, 28, 8, 30, 0),
                'status': '미흡',
                'restoration_status': '203자 / 복원 대상 87자',
                'processing_time': 414,
                'damage_level': 42.9,
                'inspection_status': '23자 완료',
                'average_reliability': 45.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 8, 36, 54)
            }
        ]
        
        for rubbing_data in seed_rubbings:
            rubbing = Rubbing(**rubbing_data)
            db.session.add(rubbing)
        
        db.session.commit()
        print(f"✅ {len(seed_rubbings)}개의 체리피킹 데이터가 추가되었습니다.")

if __name__ == '__main__':
    try:
        seed_data()
    except Exception as e:
        print(f"❌ 시드 데이터 추가 실패: {e}")
        import traceback
        traceback.print_exc()

