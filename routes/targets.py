"""
복원 대상 및 후보 관련 API 라우트
"""
from flask import Blueprint, request, jsonify, send_file
from models import db, Rubbing, RestorationTarget, Candidate, InspectionRecord
from sqlalchemy.orm import joinedload
from datetime import datetime
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
    
    # 1. 해당 탁본의 모든 Target 조회 (후보 포함, 검수 기록 포함)
    targets = RestorationTarget.query.filter_by(
        rubbing_id=rubbing_id
    ).options(
        joinedload(RestorationTarget.candidates),
        joinedload(RestorationTarget.inspection_records)
    ).all()
    
    # 2. 검수 기록을 target_id별로 매핑
    inspection_map = {}
    for target in targets:
        # 가장 최근 검수 기록만 사용
        if target.inspection_records:
            latest_record = max(target.inspection_records, key=lambda r: r.inspected_at if r.inspected_at else datetime.min)
            inspection_map[target.id] = {
                'selected_character': latest_record.selected_character,
                'selected_candidate_id': latest_record.selected_candidate_id,
                'reliability': float(latest_record.reliability) if latest_record.reliability else None
            }
    
    response_data = []
    
    for target in targets:
        # 2. 후보군 병합 로직 (Swin + MLM)
        # 한자(character)를 키로 하여 병합
        merged_candidates = {}
        # 한자별로 여러 후보가 있을 수 있으므로, 각 후보의 ID를 리스트로 저장
        candidate_ids_by_char = {}
        
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
                    "model_type": None,
                    "id": None  # 후보 ID
                }
                candidate_ids_by_char[char] = []
            
            candidate_ids_by_char[char].append(cand.id)
            
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
        
        # 각 한자에 대해 가장 우선순위가 높은 후보 ID 저장
        for char, ids in candidate_ids_by_char.items():
            if ids:
                merged_candidates[char]['id'] = ids[0]  # 첫 번째 ID 사용
        
        # 3. 계층적 우선순위 정렬 로직 (Tiered Priority)
        # 1순위: 교집합 (Intersection) - Swin과 NLP가 동시에 추천
        # 2순위: 문맥 우선 (NLP Only) - Context Match 높은 순
        # 3순위: 모양 우선 (Swin Only) - Stroke Match 높은 순
        
        tier1_intersection = []  # 교집합
        tier2_nlp_only = []      # NLP Only
        tier3_swin_only = []     # Swin Only
        
        for char, data in merged_candidates.items():
            swin = data['stroke_match']
            mlm = data['context_match']
            
            # 신뢰도 계산
            if swin is not None and mlm is not None:
                # 교집합: F1 Score 계산
                data['reliability'] = calculate_f1(swin, mlm)
                tier1_intersection.append(data)
            elif mlm is not None:
                # NLP만 있는 경우
                data['reliability'] = mlm
                tier2_nlp_only.append(data)
            elif swin is not None:
                # Swin만 있는 경우
                data['reliability'] = swin
                tier3_swin_only.append(data)
        
        # 각 그룹 정렬
        # 1순위: 교집합 - F1 Score 높은 순
        tier1_intersection.sort(key=lambda x: x['reliability'], reverse=True)
        # 2순위: NLP Only - Context Match 높은 순
        tier2_nlp_only.sort(key=lambda x: x['context_match'], reverse=True)
        # 3순위: Swin Only - Stroke Match 높은 순
        tier3_swin_only.sort(key=lambda x: x['stroke_match'], reverse=True)
        
        # 계층적 우선순위로 상위 5개 선택 (1순위가 부족하면 2순위만 사용)
        # NLP는 20개를 뽑았으므로 NLP Only가 부족할 수 없음
        top5_candidates = []
        remaining = 5
        
        # 1순위: 교집합에서 가져오기
        for candidate in tier1_intersection[:remaining]:
            top5_candidates.append(candidate)
            remaining -= 1
        
        # 2순위: NLP Only에서 가져오기 (교집합이 부족할 때만)
        # NLP는 20개를 뽑았으므로 항상 5개 이상 보장됨
        if remaining > 0:
            for candidate in tier2_nlp_only[:remaining]:
                top5_candidates.append(candidate)
                remaining -= 1
        
        # 3순위 제거: Swin Only는 제외 (NLP가 20개를 뽑았으므로 NLP Only가 부족할 수 없음)
        
        # 부족한 경우 null로 채움
        while remaining > 0:
            top5_candidates.append({
                "character": None,
                "stroke_match": None,
                "context_match": None,
                "reliability": None,
                "rank_vision": None,
                "rank_nlp": None,
                "model_type": None
            })
            remaining -= 1
        
        # 전체 후보 (시각화용) - 모든 그룹을 합쳐서 상위 10개
        all_candidates = (tier1_intersection + tier2_nlp_only + tier3_swin_only)[:10]
        
        # 5. 검수 기록 정보 추가
        inspection_info = inspection_map.get(target.id)
        
        # 6. 데이터 구조화
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
            "all_candidates": all_candidates,
            "inspection": inspection_info  # 검수 기록 정보 추가
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
    
    # 후보군 병합 로직 (get_restoration_targets와 동일, 단 후보 ID 포함)
    merged_candidates = {}
    # 한자별로 여러 후보가 있을 수 있으므로, 각 후보의 ID를 리스트로 저장
    candidate_ids_by_char = {}
    
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
                "model_type": None,
                "id": None  # 후보 ID (우선순위가 높은 것 하나만 저장)
            }
            candidate_ids_by_char[char] = []
        
        candidate_ids_by_char[char].append(cand.id)
        
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
    
    # 각 한자에 대해 가장 우선순위가 높은 후보 ID 저장
    for char, ids in candidate_ids_by_char.items():
        if ids:
            merged_candidates[char]['id'] = ids[0]  # 첫 번째 ID 사용
    
    # 계층적 우선순위 정렬 로직 (Tiered Priority)
    # 1순위: 교집합 (Intersection) - Swin과 NLP가 동시에 추천
    # 2순위: 문맥 우선 (NLP Only) - Context Match 높은 순
    # 3순위: 모양 우선 (Swin Only) - Stroke Match 높은 순
    
    tier1_intersection = []  # 교집합
    tier2_nlp_only = []      # NLP Only
    tier3_swin_only = []     # Swin Only
    
    for char, data in merged_candidates.items():
        swin = data['stroke_match']
        mlm = data['context_match']
        
        # 신뢰도 계산
        if swin is not None and mlm is not None:
            # 교집합: F1 Score 계산
            data['reliability'] = calculate_f1(swin, mlm)
            tier1_intersection.append(data)
        elif mlm is not None:
            # NLP만 있는 경우
            data['reliability'] = mlm
            tier2_nlp_only.append(data)
        elif swin is not None:
            # Swin만 있는 경우
            data['reliability'] = swin
            tier3_swin_only.append(data)
    
    # 각 그룹 정렬
    # 1순위: 교집합 - F1 Score 높은 순
    tier1_intersection.sort(key=lambda x: x['reliability'], reverse=True)
    # 2순위: NLP Only - Context Match 높은 순
    tier2_nlp_only.sort(key=lambda x: x['context_match'], reverse=True)
    # 3순위: Swin Only - Stroke Match 높은 순
    tier3_swin_only.sort(key=lambda x: x['stroke_match'], reverse=True)
    
    # 계층적 우선순위로 상위 5개 선택 (1순위가 부족하면 2순위만 사용)
    # NLP는 20개를 뽑았으므로 NLP Only가 부족할 수 없음
    top5_candidates = []
    remaining = 5
    
    # 1순위: 교집합에서 가져오기
    for candidate in tier1_intersection[:remaining]:
        top5_candidates.append(candidate)
        remaining -= 1
    
    # 2순위: NLP Only에서 가져오기 (교집합이 부족할 때만)
    # NLP는 20개를 뽑았으므로 항상 5개 이상 보장됨
    if remaining > 0:
        for candidate in tier2_nlp_only[:remaining]:
            top5_candidates.append(candidate)
            remaining -= 1
    
    # 3순위 제거: Swin Only는 제외 (NLP가 20개를 뽑았으므로 NLP Only가 부족할 수 없음)
    
    # 부족한 경우 null로 채움
    while remaining > 0:
        top5_candidates.append({
            "character": None,
            "stroke_match": None,
            "context_match": None,
            "reliability": None,
            "rank_vision": None,
            "rank_nlp": None,
            "model_type": None
        })
        remaining -= 1
    
    # 전체 후보 (시각화용) - 모든 그룹을 합쳐서 상위 10개
    all_candidates = (tier1_intersection + tier2_nlp_only + tier3_swin_only)[:10]
    
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

