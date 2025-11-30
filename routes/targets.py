"""
복원 대상 및 후보 관련 API 라우트
"""
from flask import Blueprint, request, jsonify, send_file
from models import db, Rubbing, RestorationTarget, Candidate
import os

targets_bp = Blueprint('targets', __name__)


@targets_bp.route('/api/rubbings/<int:rubbing_id>/restoration-targets', methods=['GET'])
def get_restoration_targets(rubbing_id):
    """복원 대상 목록 조회"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    targets = RestorationTarget.query.filter_by(rubbing_id=rubbing_id).all()
    
    return jsonify([target.to_dict() for target in targets])


@targets_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/candidates', methods=['GET'])
def get_candidates(rubbing_id, target_id):
    """후보 한자 목록 조회 (교집합 처리 포함)"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    target = RestorationTarget.query.filter_by(
        id=target_id,
        rubbing_id=rubbing_id
    ).first()
    
    if not target:
        return jsonify({'error': 'Target not found'}), 404
    
    # 전체 후보 조회
    all_candidates = Candidate.query.filter_by(target_id=target_id).all()
    
    # 교집합 계산: 획 일치도와 문맥 일치도 둘 다 존재하는 후보
    intersection_candidates = [
        c for c in all_candidates
        if c.stroke_match is not None and c.context_match is not None
    ]
    
    # 신뢰도 기준으로 정렬
    sorted_intersection = sorted(
        intersection_candidates,
        key=lambda c: float(c.reliability) if c.reliability is not None else 0,
        reverse=True
    )
    
    # 상위 5개 선택, 부족하면 null로 채움
    top5_candidates = []
    for i in range(5):
        if i < len(sorted_intersection):
            candidate = sorted_intersection[i]
            top5_candidates.append(candidate.to_dict())
        else:
            # null 값으로 채움
            top5_candidates.append({
                'id': None,
                'character': None,
                'stroke_match': None,
                'context_match': None,
                'rank_vision': None,
                'rank_nlp': None,
                'model_type': None,
                'reliability': None
            })
    
    return jsonify({
        'top5': top5_candidates,
        'all': [c.to_dict() for c in all_candidates]
    })


@targets_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/reasoning', methods=['GET'])
def get_reasoning(rubbing_id, target_id):
    """유추 근거 데이터 조회"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    target = RestorationTarget.query.filter_by(
        id=target_id,
        rubbing_id=rubbing_id
    ).first()
    
    if not target:
        return jsonify({'error': 'Target not found'}), 404
    
    # 전체 후보 조회
    all_candidates = Candidate.query.filter_by(target_id=target_id).all()
    
    # Vision 모델 후보 (획 일치도 기준 정렬)
    vision_candidates = sorted(
        [c for c in all_candidates if c.stroke_match is not None],
        key=lambda c: float(c.stroke_match) if c.stroke_match is not None else 0,
        reverse=True
    )[:10]  # 상위 10개
    
    # NLP 모델 후보 (문맥 일치도 기준 정렬)
    nlp_candidates = sorted(
        [c for c in all_candidates],
        key=lambda c: float(c.context_match) if c.context_match is not None else 0,
        reverse=True
    )[:10]  # 상위 10개
    
    return jsonify({
        'imgUrl': target.cropped_image_url if target.cropped_image_url else None,
        'vision': [c.to_dict() for c in vision_candidates],
        'nlp': [c.to_dict() for c in nlp_candidates]
    })


@targets_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/cropped-image', methods=['GET'])
def get_cropped_image(rubbing_id, target_id):
    """복원 대상 글자 크롭 이미지 조회"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    target = RestorationTarget.query.filter_by(
        id=target_id,
        rubbing_id=rubbing_id
    ).first()
    
    if not target:
        return jsonify({'error': 'Target not found'}), 404
    
    if not target.cropped_image_url:
        return jsonify({'error': 'Cropped image not found'}), 404
    
    # 크롭된 이미지 경로 (상대 경로를 절대 경로로 변환)
    cropped_image_path = os.path.join(os.getcwd(), target.cropped_image_url.lstrip('/'))
    
    if not os.path.exists(cropped_image_path):
        return jsonify({'error': 'Cropped image file not found'}), 404
    
    return send_file(cropped_image_path, mimetype='image/jpeg')

