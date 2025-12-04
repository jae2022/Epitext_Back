"""
복원 대상 및 후보 관련 API 라우트
"""
from flask import Blueprint, request, jsonify, send_file
from models import db, Rubbing, RestorationTarget, Candidate, InspectionRecord
from sqlalchemy.orm import joinedload
import os
import logging

logger = logging.getLogger(__name__)

targets_bp = Blueprint('targets', __name__)


def calculate_f1(score1, score2):
    """두 점수의 조화 평균(F1 Score 유사 방식) 계산"""
    if score1 is None or score2 is None:
        return 0.0
    if score1 + score2 == 0:
        return 0.0
    return 2 * (score1 * score2) / (score1 + score2)


@targets_bp.route('/api/rubbings/<int:rubbing_id>/restoration-targets', methods=['GET'])
def get_restoration_targets(rubbing_id):
    """
    특정 탁본의 모든 복원 대상(Target)과 후보(Candidate)를 병합하여 반환
    프론트엔드 요구사항에 맞게 Swin과 MLM 결과를 병합하고 F1 Score를 계산
    """
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    # 1. 해당 탁본의 모든 Target 조회 (후보 포함)
    targets = RestorationTarget.query.filter_by(
        rubbing_id=rubbing_id
    ).options(joinedload(RestorationTarget.candidates)).all()
    
    response_data = []
    
    for target in targets:
        # 2. 후보군 병합 로직 (Swin + MLM)
        # 한자(character)를 키로 하여 병합
        merged_candidates = {}
        
        for cand in target.candidates:
            char = cand.character
            if char not in merged_candidates:
                merged_candidates[char] = {
                    "character": char,
                    "stroke_match": None,  # Swin Score
                    "context_match": None,  # MLM Score
                    "reliability": 0.0,
                    "rank_vision": None,
                    "rank_nlp": None,
                    "model_type": None
                }
            
            # 점수 매핑
            if cand.stroke_match is not None:
                merged_candidates[char]['stroke_match'] = float(cand.stroke_match)
                merged_candidates[char]['rank_vision'] = cand.rank_vision
            if cand.context_match is not None:
                merged_candidates[char]['context_match'] = float(cand.context_match)
                merged_candidates[char]['rank_nlp'] = cand.rank_nlp
            
            # model_type 설정
            if merged_candidates[char]['stroke_match'] is not None and merged_candidates[char]['context_match'] is not None:
                merged_candidates[char]['model_type'] = 'both'
            elif merged_candidates[char]['stroke_match'] is not None:
                merged_candidates[char]['model_type'] = 'vision'
            elif merged_candidates[char]['context_match'] is not None:
                merged_candidates[char]['model_type'] = 'nlp'
        
        # 3. 신뢰도(F1) 계산 및 정렬 로직 적용
        final_candidates = []
        for char, data in merged_candidates.items():
            swin = data['stroke_match']
            mlm = data['context_match']
            
            # 신뢰도 계산
            if swin is not None and mlm is not None:
                # 교집합: F1 Score 계산
                data['reliability'] = calculate_f1(swin, mlm)
            elif mlm is not None:
                # MLM만 있는 경우
                data['reliability'] = mlm
            elif swin is not None:
                # Swin만 있는 경우
                data['reliability'] = swin
            
            final_candidates.append(data)
        
        # 4. 정렬 (프론트엔드 요구사항: 교집합 F1순 -> 그 외 점수순)
        # 교집합 여부(swin & mlm 둘 다 있음)를 우선순위로 둠
        final_candidates.sort(
            key=lambda x: (
                1 if (x['stroke_match'] is not None and x['context_match'] is not None) else 0,  # 교집합 우선
                x['reliability']  # 점수 높은 순
            ),
            reverse=True
        )
        
        # 상위 5개만 유지 (부족하면 null로 채움)
        top5_candidates = []
        for i in range(5):
            if i < len(final_candidates):
                top5_candidates.append(final_candidates[i])
            else:
                # null 값으로 채움
                top5_candidates.append({
                    "character": None,
                    "stroke_match": None,
                    "context_match": None,
                    "reliability": None,
                    "rank_vision": None,
                    "rank_nlp": None,
                    "model_type": None
                })
        
        # 전체 후보 (시각화용)
        all_candidates = final_candidates[:10]  # 상위 10개
        
        # 5. 데이터 구조화
        response_data.append({
            "id": target.id,
            "row_index": target.row_index,
            "char_index": target.char_index,
            "position": target.position,
            "damage_type": target.damage_type,
            "cropped_image_url": target.cropped_image_url,
            "crop_x": target.crop_x,
            "crop_y": target.crop_y,
            "crop_width": target.crop_width,
            "crop_height": target.crop_height,
            "candidates": top5_candidates,
            "all_candidates": all_candidates
        })
    
    return jsonify(response_data)


@targets_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/candidates', methods=['GET'])
def get_candidates(rubbing_id, target_id):
    """후보 한자 목록 조회 (교집합 처리 포함) - get_restoration_targets와 동일한 로직 사용"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    target = RestorationTarget.query.filter_by(
        id=target_id,
        rubbing_id=rubbing_id
    ).options(joinedload(RestorationTarget.candidates)).first()
    
    if not target:
        return jsonify({'error': 'Target not found'}), 404
    
    # 후보군 병합 로직 (get_restoration_targets와 동일)
    merged_candidates = {}
    
    for cand in target.candidates:
        char = cand.character
        if char not in merged_candidates:
            merged_candidates[char] = {
                "character": char,
                "stroke_match": None,
                "context_match": None,
                "reliability": 0.0,
                "rank_vision": None,
                "rank_nlp": None,
                "model_type": None
            }
        
        if cand.stroke_match is not None:
            merged_candidates[char]['stroke_match'] = float(cand.stroke_match)
            merged_candidates[char]['rank_vision'] = cand.rank_vision
        if cand.context_match is not None:
            merged_candidates[char]['context_match'] = float(cand.context_match)
            merged_candidates[char]['rank_nlp'] = cand.rank_nlp
        
        if merged_candidates[char]['stroke_match'] is not None and merged_candidates[char]['context_match'] is not None:
            merged_candidates[char]['model_type'] = 'both'
        elif merged_candidates[char]['stroke_match'] is not None:
            merged_candidates[char]['model_type'] = 'vision'
        elif merged_candidates[char]['context_match'] is not None:
            merged_candidates[char]['model_type'] = 'nlp'
    
    # 신뢰도 계산
    final_candidates = []
    for char, data in merged_candidates.items():
        swin = data['stroke_match']
        mlm = data['context_match']
        
        if swin is not None and mlm is not None:
            data['reliability'] = calculate_f1(swin, mlm)
        elif mlm is not None:
            data['reliability'] = mlm
        elif swin is not None:
            data['reliability'] = swin
        
        final_candidates.append(data)
    
    # 정렬
    final_candidates.sort(
        key=lambda x: (
            1 if (x['stroke_match'] is not None and x['context_match'] is not None) else 0,
            x['reliability']
        ),
        reverse=True
    )
    
    # 상위 5개
    top5_candidates = []
    for i in range(5):
        if i < len(final_candidates):
            top5_candidates.append(final_candidates[i])
        else:
            top5_candidates.append({
                "character": None,
                "stroke_match": None,
                "context_match": None,
                "reliability": None,
                "rank_vision": None,
                "rank_nlp": None,
                "model_type": None
            })
    
    # 전체 후보 (상위 10개)
    all_candidates = final_candidates[:10]
    
    return jsonify({
        'top5': top5_candidates,
        'all': all_candidates
    })


@targets_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/reasoning', methods=['GET'])
def get_reasoning(rubbing_id, target_id):
    """유추 근거 데이터 조회 - ReasoningCluster용"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    target = RestorationTarget.query.filter_by(
        id=target_id,
        rubbing_id=rubbing_id
    ).options(joinedload(RestorationTarget.candidates)).first()
    
    if not target:
        return jsonify({'error': 'Target not found'}), 404
    
    # 전체 후보 조회 및 병합
    merged_candidates = {}
    
    for cand in target.candidates:
        char = cand.character
        if char not in merged_candidates:
            merged_candidates[char] = {
                "character": char,
                "stroke_match": None,
                "context_match": None
            }
        
        if cand.stroke_match is not None:
            merged_candidates[char]['stroke_match'] = float(cand.stroke_match)
        if cand.context_match is not None:
            merged_candidates[char]['context_match'] = float(cand.context_match)
    
    # Vision 모델 후보 (획 일치도 기준 정렬)
    vision_candidates = [
        {
            "character": char,
            "stroke_match": data['stroke_match'],
            "score": data['stroke_match'] / 100 if data['stroke_match'] is not None else 0
        }
        for char, data in merged_candidates.items()
        if data['stroke_match'] is not None
    ]
    vision_candidates.sort(key=lambda x: x['stroke_match'], reverse=True)
    vision_candidates = vision_candidates[:10]
    
    # NLP 모델 후보 (문맥 일치도 기준 정렬)
    nlp_candidates = [
        {
            "character": char,
            "context_match": data['context_match'],
            "score": data['context_match'] / 100 if data['context_match'] is not None else 0
        }
        for char, data in merged_candidates.items()
        if data['context_match'] is not None
    ]
    nlp_candidates.sort(key=lambda x: x['context_match'], reverse=True)
    nlp_candidates = nlp_candidates[:10]
    
    return jsonify({
        'imgUrl': target.cropped_image_url if target.cropped_image_url else None,
        'vision': vision_candidates,
        'nlp': nlp_candidates
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

