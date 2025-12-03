"""
프론트엔드 더미 데이터를 DB에 시드하는 스크립트
프론트엔드의 mockRubbingList와 동일한 데이터를 생성합니다.
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models import db, Rubbing, RubbingDetail, RubbingStatistics, RestorationTarget, Candidate, InspectionRecord
from datetime import datetime
import json

def delete_all_data():
    """모든 데이터 삭제"""
    app = create_app()
    with app.app_context():
        InspectionRecord.query.delete()
        Candidate.query.delete()
        RestorationTarget.query.delete()
        RubbingStatistics.query.delete()
        RubbingDetail.query.delete()
        Rubbing.query.delete()
        db.session.commit()
        print("✅ 기존 데이터 삭제 완료")

def seed_data(force=False):
    """프론트엔드 mockRubbingList와 동일한 데이터 생성
    
    Args:
        force: True이면 기존 데이터를 자동으로 삭제하고 새로 생성
    """
    app = create_app()
    
    with app.app_context():
        # 기존 데이터 확인
        existing_count = Rubbing.query.count()
        if existing_count > 0:
            if force:
                print(f"⚠️  기존 {existing_count}개의 탁본 데이터를 삭제하고 새로 생성합니다.")
                delete_all_data()
            else:
                print(f"⚠️  이미 {existing_count}개의 탁본 데이터가 존재합니다.")
                print("   기존 데이터를 삭제하려면: python database/seed_data.py --force")
                return
        
        # 프론트엔드 mockRubbingList와 동일한 데이터
        seed_rubbings = [
            {
                'id': 8,
                'image_url': '/images/rubbings/rubbing_8.jpg',
                'filename': '귀법사적수화현응모지명_8.jpg',
                'created_at': datetime(2025, 10, 28, 10, 0, 0),
                'status': '처리중',
                'restoration_status': None,
                'processing_time': None,
                'damage_level': None,
                'inspection_status': None,
                'average_reliability': None,
                'is_completed': False,
                'processed_at': None
            },
            {
                'id': 7,
                'image_url': '/images/rubbings/rubbing_7.jpg',
                'filename': '귀법사적수화현응모지명_7.jpg',
                'created_at': datetime(2025, 10, 28, 9, 30, 0),
                'status': '우수',
                'restoration_status': '356자 / 복원 대상 23자',
                'processing_time': 222,
                'damage_level': 6.5,
                'inspection_status': '12자 완료',
                'average_reliability': 92.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 9, 33, 42)
            },
            {
                'id': 6,
                'image_url': '/images/rubbings/rubbing_6.jpg',
                'filename': '귀법사적수화현응모지명_6.jpg',
                'created_at': datetime(2025, 10, 28, 9, 0, 0),
                'status': '양호',
                'restoration_status': '68자 / 복원 대상 12자',
                'processing_time': 201,
                'damage_level': 17.6,
                'inspection_status': '12자 완료',
                'average_reliability': 76.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 9, 3, 21)
            },
            {
                'id': 5,
                'image_url': '/images/rubbings/rubbing_5.jpg',
                'filename': '귀법사적수화현응모지명_5.jpg',
                'created_at': datetime(2025, 10, 28, 8, 30, 0),
                'status': '우수',
                'restoration_status': '112자 / 복원 대상 8자',
                'processing_time': 225,
                'damage_level': 7.1,
                'inspection_status': '5자 완료',
                'average_reliability': 92.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 8, 33, 45)
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
                'id': 3,
                'image_url': '/images/rubbings/rubbing_3.jpg',
                'filename': '귀법사적수화현응모지명_3.jpg',
                'created_at': datetime(2025, 10, 28, 7, 30, 0),
                'status': '양호',
                'restoration_status': '15자 / 복원 대상 8자',
                'processing_time': 137,
                'damage_level': 53.3,
                'inspection_status': '2자 완료',
                'average_reliability': 71.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 7, 32, 17)
            },
            {
                'id': 2,
                'image_url': '/images/rubbings/rubbing_2.jpg',
                'filename': '귀법사적수화현응모지명_2.jpg',
                'created_at': datetime(2025, 10, 28, 7, 0, 0),
                'status': '미흡',
                'restoration_status': '203자 / 복원 대상 87자',
                'processing_time': 414,
                'damage_level': 42.9,
                'inspection_status': '23자 완료',
                'average_reliability': 45.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 7, 6, 54)
            },
            {
                'id': 1,
                'image_url': '/images/rubbings/rubbing_1.jpg',
                'filename': '귀법사적수화현응모지명.jpg',
                'created_at': datetime(2025, 10, 28, 6, 30, 0),
                'status': '미흡',
                'restoration_status': '47자 / 복원 대상 29자',
                'processing_time': 273,
                'damage_level': 61.7,
                'inspection_status': '14자 완료',
                'average_reliability': 52.0,
                'is_completed': False,
                'processed_at': datetime(2025, 10, 28, 6, 34, 33)
            },
        ]
        
        # 각 탁본에 대한 상세 정보 생성
        detail_data = {
            8: {
                'text_content': json.dumps(["처리중인 탁본입니다."], ensure_ascii=False),
                'text_content_with_punctuation': json.dumps(["처리중인 탁본입니다."], ensure_ascii=False),
                'font_types': json.dumps(["행서체"], ensure_ascii=False),
                'damage_percentage': None,
                'total_processing_time': None,
            },
            7: {
                'text_content': json.dumps(["우수 상태의 탁본입니다."] * 10, ensure_ascii=False),
                'text_content_with_punctuation': json.dumps(["우수 상태의 탁본입니다。"] * 10, ensure_ascii=False),
                'font_types': json.dumps(["행서체", "전서체"], ensure_ascii=False),
                'damage_percentage': 6.5,
                'total_processing_time': 222,
            },
            6: {
                'text_content': json.dumps(["양호 상태의 탁본입니다."] * 5, ensure_ascii=False),
                'text_content_with_punctuation': json.dumps(["양호 상태의 탁본입니다。"] * 5, ensure_ascii=False),
                'font_types': json.dumps(["행서체"], ensure_ascii=False),
                'damage_percentage': 17.6,
                'total_processing_time': 201,
            },
            5: {
                'text_content': json.dumps(["우수 상태의 탁본입니다."] * 8, ensure_ascii=False),
                'text_content_with_punctuation': json.dumps(["우수 상태의 탁본입니다。"] * 8, ensure_ascii=False),
                'font_types': json.dumps(["전서체"], ensure_ascii=False),
                'damage_percentage': 7.1,
                'total_processing_time': 225,
            },
            4: {
                'text_content': json.dumps(["미흡 상태의 탁본입니다."] * 6, ensure_ascii=False),
                'text_content_with_punctuation': json.dumps(["미흡 상태의 탁본입니다。"] * 6, ensure_ascii=False),
                'font_types': json.dumps(["행서체", "전서체"], ensure_ascii=False),
                'damage_percentage': 34.8,
                'total_processing_time': 302,
            },
            3: {
                'text_content': json.dumps(["양호 상태의 탁본입니다."] * 3, ensure_ascii=False),
                'text_content_with_punctuation': json.dumps(["양호 상태의 탁본입니다。"] * 3, ensure_ascii=False),
                'font_types': json.dumps(["행서체"], ensure_ascii=False),
                'damage_percentage': 53.3,
                'total_processing_time': 137,
            },
            2: {
                'text_content': json.dumps(["미흡 상태의 탁본입니다."] * 12, ensure_ascii=False),
                'text_content_with_punctuation': json.dumps(["미흡 상태의 탁본입니다。"] * 12, ensure_ascii=False),
                'font_types': json.dumps(["전서체"], ensure_ascii=False),
                'damage_percentage': 42.9,
                'total_processing_time': 414,
            },
            1: {
                'text_content': json.dumps([
                    "高□洛□歸法寺住持",
                    "見性寂炤首□玄應者",
                    "立□第十五□肅宗□子",
                    "□□歲下元己未二月十",
                    "□日甲子薨卒二十一日",
                    "壬申茶□以三月□五日",
                    "乙酉□舍利□於八德□",
                ], ensure_ascii=False),
                'text_content_with_punctuation': json.dumps([
                    "高□洛□歸法寺住持，",
                    "見性寂炤首□玄應者。",
                    "立□第十五□肅宗□子，",
                    "□□歲下元己未二月十。",
                    "□日甲子薨卒二十一日，",
                    "壬申茶□以三月□五日。",
                    "乙酉□舍利□於八德□，",
                ], ensure_ascii=False),
                'font_types': json.dumps(["행서체", "전서체"], ensure_ascii=False),
                'damage_percentage': 61.7,
                'total_processing_time': 273,
            },
        }
        
        # 통계 데이터 (복원 대상 수 기반)
        statistics_data = {
            8: {'total_characters': 0, 'restoration_targets': 0, 'partial_damage': 0, 'complete_damage': 0, 'restoration_percentage': 0.0},
            7: {'total_characters': 356, 'restoration_targets': 23, 'partial_damage': 14, 'complete_damage': 9, 'restoration_percentage': 6.5},
            6: {'total_characters': 68, 'restoration_targets': 12, 'partial_damage': 7, 'complete_damage': 5, 'restoration_percentage': 17.6},
            5: {'total_characters': 112, 'restoration_targets': 8, 'partial_damage': 5, 'complete_damage': 3, 'restoration_percentage': 7.1},
            4: {'total_characters': 89, 'restoration_targets': 31, 'partial_damage': 19, 'complete_damage': 12, 'restoration_percentage': 34.8},
            3: {'total_characters': 15, 'restoration_targets': 8, 'partial_damage': 5, 'complete_damage': 3, 'restoration_percentage': 53.3},
            2: {'total_characters': 203, 'restoration_targets': 87, 'partial_damage': 52, 'complete_damage': 35, 'restoration_percentage': 42.9},
            1: {'total_characters': 47, 'restoration_targets': 29, 'partial_damage': 18, 'complete_damage': 11, 'restoration_percentage': 61.7},
        }
        
        # 탁본 생성
        for rubbing_data in seed_rubbings:
            rubbing = Rubbing(**rubbing_data)
            db.session.add(rubbing)
            db.session.flush()  # ID를 얻기 위해 flush
            
            rubbing_id = rubbing.id
            
            # RubbingDetail 생성
            if rubbing_id in detail_data:
                detail = RubbingDetail(
                    rubbing_id=rubbing_id,
                    **detail_data[rubbing_id]
                )
                db.session.add(detail)
            
            # RubbingStatistics 생성
            if rubbing_id in statistics_data:
                stats = RubbingStatistics(
                    rubbing_id=rubbing_id,
                    **statistics_data[rubbing_id]
                )
                db.session.add(stats)
            
            # RestorationTarget 생성 (id=1만 상세하게)
            if rubbing_id == 1:
                # text_content_with_punctuation에서 □ 위치 찾기
                text_lines = detail_data[1]['text_content_with_punctuation']
                text_lines = json.loads(text_lines) if isinstance(text_lines, str) else text_lines
                
                target_id = 1
                for row_idx, line in enumerate(text_lines):
                    for char_idx, char in enumerate(line):
                        if char == '□':
                            damage_type = '부분_훼손' if target_id % 2 == 0 else '완전_훼손'
                            target = RestorationTarget(
                                rubbing_id=rubbing_id,
                                row_index=row_idx,
                                char_index=char_idx,
                                position=f'{row_idx + 1}행 {char_idx + 1}자',
                                damage_type=damage_type
                            )
                            db.session.add(target)
                            db.session.flush()
                            
                            # Candidate 생성 (각 target마다 5개씩)
                            candidates_data = [
                                {'character': '麗', 'stroke_match': 85.4 if damage_type == '부분_훼손' else None, 'context_match': 76.8, 'rank_vision': 1 if damage_type == '부분_훼손' else None, 'rank_nlp': 1},
                                {'character': '郡', 'stroke_match': 55.8 if damage_type == '부분_훼손' else None, 'context_match': 68.5, 'rank_vision': 2 if damage_type == '부분_훼손' else None, 'rank_nlp': 2},
                                {'character': '鄕', 'stroke_match': 50.4 if damage_type == '부분_훼손' else None, 'context_match': 65.2, 'rank_vision': 3 if damage_type == '부분_훼손' else None, 'rank_nlp': 3},
                                {'character': '麓', 'stroke_match': 45.2 if damage_type == '부분_훼손' else None, 'context_match': 62.1, 'rank_vision': 4 if damage_type == '부분_훼손' else None, 'rank_nlp': 4},
                                {'character': '楚', 'stroke_match': 40.1 if damage_type == '부분_훼손' else None, 'context_match': 58.9, 'rank_vision': 5 if damage_type == '부분_훼손' else None, 'rank_nlp': 5},
                            ]
                            
                            for cand_data in candidates_data:
                                # model_type 계산
                                if cand_data['stroke_match'] is not None and cand_data['context_match'] is not None:
                                    model_type = 'both'
                                elif cand_data['stroke_match'] is not None:
                                    model_type = 'vision'
                                else:
                                    model_type = 'nlp'
                                
                                # reliability 계산 (F1 Score)
                                stroke = cand_data['stroke_match'] if cand_data['stroke_match'] is not None else 0
                                context = cand_data['context_match']
                                if stroke == 0 and context == 0:
                                    reliability = 0
                                elif stroke == 0:
                                    reliability = context
                                else:
                                    reliability = (2 * stroke * context) / (stroke + context)
                                
                                candidate = Candidate(
                                    target_id=target.id,
                                    model_type=model_type,
                                    reliability=round(reliability, 1),
                                    **cand_data
                                )
                                db.session.add(candidate)
                            
                            target_id += 1
            else:
                # 다른 탁본들은 간단하게 target만 생성
                target_count = statistics_data.get(rubbing_id, {}).get('restoration_targets', 0)
                for i in range(target_count):
                    damage_type = '부분_훼손' if i % 2 == 0 else '완전_훼손'
                    row_idx = i // 10
                    char_idx = i % 10
                    target = RestorationTarget(
                        rubbing_id=rubbing_id,
                        row_index=row_idx,
                        char_index=char_idx,
                        position=f'{row_idx + 1}행 {char_idx + 1}자',
                        damage_type=damage_type
                    )
                    db.session.add(target)
                    db.session.flush()
                    
                    # 간단한 candidate 생성
                    for j in range(5):
                        stroke_match = 80.0 - j*5 if damage_type == '부분_훼손' else None
                        context_match = 70.0 - j*5
                        
                        # model_type 계산
                        if stroke_match is not None:
                            model_type = 'both'
                        else:
                            model_type = 'nlp'
                        
                        # reliability 계산
                        stroke = stroke_match if stroke_match is not None else 0
                        if stroke == 0 and context_match == 0:
                            reliability = 0
                        elif stroke == 0:
                            reliability = context_match
                        else:
                            reliability = (2 * stroke * context_match) / (stroke + context_match)
                        
                        candidate = Candidate(
                            target_id=target.id,
                            character=f'候補{j+1}',
                            stroke_match=stroke_match,
                            context_match=context_match,
                            rank_vision=j+1 if damage_type == '부분_훼손' else None,
                            rank_nlp=j+1,
                            model_type=model_type,
                            reliability=round(reliability, 1)
                        )
                        db.session.add(candidate)
        
        # InspectionRecord 생성 (검수 완료된 글자들)
        inspection_data = {
            7: 12,  # 12자 완료
            6: 12,  # 12자 완료
            5: 5,   # 5자 완료
            4: 31,  # 31자 완료
            3: 2,   # 2자 완료
            2: 23,  # 23자 완료
            1: 14,  # 14자 완료
        }
        
        for rubbing_id, inspected_count in inspection_data.items():
            rubbing = Rubbing.query.get(rubbing_id)
            if not rubbing:
                continue
            
            targets = RestorationTarget.query.filter_by(rubbing_id=rubbing_id).limit(inspected_count).all()
            for target in targets:
                # 첫 번째 candidate 선택
                candidate = Candidate.query.filter_by(target_id=target.id).order_by(Candidate.reliability.desc()).first()
                if candidate:
                    inspection = InspectionRecord(
                        rubbing_id=rubbing_id,
                        target_id=target.id,
                        selected_character=candidate.character,
                        selected_candidate_id=candidate.id,
                        reliability=candidate.reliability,
                        inspected_at=rubbing.processed_at if rubbing.processed_at else datetime.now()
                    )
                    db.session.add(inspection)
        
        db.session.commit()
        print(f"✅ {len(seed_rubbings)}개의 탁본 데이터가 생성되었습니다.")
        print(f"   - Rubbing: {len(seed_rubbings)}개")
        print(f"   - RubbingDetail: {len([d for d in detail_data.keys()])}개")
        print(f"   - RubbingStatistics: {len([s for s in statistics_data.keys()])}개")
        
        # 생성된 데이터 통계
        total_targets = RestorationTarget.query.count()
        total_candidates = Candidate.query.count()
        total_inspections = InspectionRecord.query.count()
        print(f"   - RestorationTarget: {total_targets}개")
        print(f"   - Candidate: {total_candidates}개")
        print(f"   - InspectionRecord: {total_inspections}개")

if __name__ == '__main__':
    import sys
    force = '--force' in sys.argv or '-f' in sys.argv
    try:
        seed_data(force=force)
    except Exception as e:
        print(f"❌ 시드 데이터 생성 실패: {e}")
        import traceback
        traceback.print_exc()
