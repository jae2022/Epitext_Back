"""
검수 관련 API 라우트
"""
from flask import Blueprint, request, jsonify
from models import db, Rubbing, RestorationTarget, Candidate, InspectionRecord
from sqlalchemy import func
from datetime import datetime

inspection_bp = Blueprint('inspection', __name__)


@inspection_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/inspect', methods=['POST'])
def inspect_target(rubbing_id, target_id):
    """검수 결과 저장"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    target = RestorationTarget.query.filter_by(
        id=target_id,
        rubbing_id=rubbing_id
    ).first()
    
    if not target:
        return jsonify({'error': 'Target not found'}), 404
    
    data = request.get_json()
    selected_character = data.get('selected_character')
    selected_candidate_id = data.get('selected_candidate_id')
    
    if not selected_character:
        return jsonify({'error': 'selected_character is required'}), 400
    
    # 선택된 후보의 신뢰도 조회
    reliability = None
    if selected_candidate_id:
        candidate = Candidate.query.get(selected_candidate_id)
        if candidate:
            reliability = float(candidate.reliability) if candidate.reliability is not None else None
    
    # 검수 기록 저장
    inspection_record = InspectionRecord(
        rubbing_id=rubbing_id,
        target_id=target_id,
        selected_character=selected_character,
        selected_candidate_id=selected_candidate_id,
        reliability=reliability
    )
    
    db.session.add(inspection_record)
    
    # 검수 현황 업데이트
    inspected_count = InspectionRecord.query.filter_by(rubbing_id=rubbing_id).count()
    rubbing.inspection_status = f"{inspected_count}자 완료"
    
    # 평균 신뢰도 계산
    avg_reliability = db.session.query(func.avg(InspectionRecord.reliability)).filter_by(
        rubbing_id=rubbing_id
    ).scalar()
    
    if avg_reliability is not None:
        rubbing.average_reliability = float(avg_reliability)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'inspected_at': inspection_record.inspected_at.isoformat() if inspection_record.inspected_at else None
    })

