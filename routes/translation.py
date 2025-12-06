# -*- coding: utf-8 -*-
"""
번역 API 라우트
"""
from flask import Blueprint, request, jsonify
import logging
import json
from ai_modules.translation_engine import get_translation_engine
from models import RubbingDetail, RestorationTarget, InspectionRecord

logger = logging.getLogger(__name__)

translation_bp = Blueprint('translation', __name__)

@translation_bp.route('/api/translation', methods=['POST'])
def translate_text():
    """
    한문 텍스트 번역 API
    
    Request Body:
        {
            "text": "번역할 한문 텍스트"
        }
    
    Response:
        {
            "success": true,
            "reading": "음독 결과",
            "entities": "고유명사 추출 결과",
            "translation": "최종 번역 결과",
            "model": "사용된 모델명"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        logger.info(f"[TRANSLATION-API] 번역 요청: {text[:50]}...")
        
        # 번역 엔진 호출
        engine = get_translation_engine()
        result = engine.translate(text)
        
        if result.get('success'):
            logger.info(f"[TRANSLATION-API] 번역 성공: 모델={result.get('model')}")
            return jsonify(result), 200
        else:
            error_msg = result.get('error', 'Translation failed')
            logger.error(f"[TRANSLATION-API] 번역 실패: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        logger.error(f"[TRANSLATION-API] 예외 발생: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# -------------------------------------------------------------------------
# [Helper] 정확한 줄 찾기 및 치환 로직
# -------------------------------------------------------------------------
def find_target_line_and_replace(rubbing_id, target_id, replacement_char=None):
    """
    구두점이 포함된 텍스트에서 target_id가 위치한 '정확한 줄'을 찾고,
    1) 이미 검수된 글자들(InspectionRecord)을 해당 위치의 □에 채워 넣고
    2) 현재 보고 있는 target_id에 대해서는 replacement_char(미리보기 글자)로 치환하여 반환합니다.
    """
    # 1. 해당 탁본의 모든 Target을 순서대로 조회 (row_index, char_index 순서로 정렬)
    all_targets = RestorationTarget.query.filter_by(rubbing_id=rubbing_id).order_by(
        RestorationTarget.row_index, RestorationTarget.char_index
    ).all()
    
    # 2. 이미 검수 완료된 기록 조회 (DB에서 가져옴) [추가된 로직]
    # 각 Target에 대해 가장 최근 검수 기록만 가져오기
    filled_map = {}
    for target in all_targets:
        latest_insp = InspectionRecord.query.filter_by(
            rubbing_id=rubbing_id,
            target_id=target.id
        ).order_by(InspectionRecord.inspected_at.desc()).first()
        if latest_insp:
            filled_map[target.id] = latest_insp.selected_character
    
    # 3. 현재 Target이 전체 중에서 몇 번째 □인지 찾기 (Global Index)
    try:
        target_ids = [t.id for t in all_targets]
        target_global_index = target_ids.index(target_id)
    except ValueError:
        return None, "Target not found"

    # 4. 구두점 포함 텍스트 가져오기
    detail = RubbingDetail.query.filter_by(rubbing_id=rubbing_id).first()
    if not detail or not detail.text_content_with_punctuation:
        return None, "Text content not found"
        
    try:
        text_lines = json.loads(detail.text_content_with_punctuation)
    except:
        text_lines = []

    # 5. 줄을 순회하며 현재 Target이 포함된 줄 찾기
    current_mask_count = 0
    found_line_index = -1
    
    # 해당 줄의 시작 마스크 인덱스 (Global index 기준)
    line_start_mask_index = 0
    
    for idx, line in enumerate(text_lines):
        line_masks = line.count('□')
        if current_mask_count <= target_global_index < (current_mask_count + line_masks):
            found_line_index = idx
            line_start_mask_index = current_mask_count
            break
        current_mask_count += line_masks
        
    if found_line_index == -1:
        return None, "Target line mismatch"

    # 6. 찾은 줄 가져오기
    original_line = text_lines[found_line_index]
    
    # 7. [핵심 수정] 줄 내의 모든 □를 순회하며 적절한 글자로 치환
    #    - 현재 미리보기 중인 글자 (replacement_char)
    #    - 이미 검수 완료된 글자 (filled_map)
    
    chars = list(original_line)
    seen_masks_in_line = 0
    char_index_in_line = -1  # 현재 선택된 글자의 줄 내 인덱스 (하이라이팅용)

    for i, char in enumerate(chars):
        if char == '□':
            # 이 □의 전체(Global) 순번 계산
            current_mask_global_idx = line_start_mask_index + seen_masks_in_line
            
            # 유효성 검사
            if current_mask_global_idx < len(all_targets):
                t_obj = all_targets[current_mask_global_idx]
                
                # Case A: 현재 팝업에서 보고 있는 Target인 경우
                if t_obj.id == target_id:
                    if replacement_char:  # 미리보기 글자가 있으면 치환
                        chars[i] = replacement_char
                    char_index_in_line = i  # 인덱스 저장 (나중에 하이라이팅)
                
                # Case B: 이미 검수가 끝난 다른 Target인 경우 [추가됨]
                elif t_obj.id in filled_map:
                    chars[i] = filled_map[t_obj.id]  # DB에 저장된 검수 글자로 치환
            
            seen_masks_in_line += 1
            
    final_text = "".join(chars)
        
    return final_text, char_index_in_line


# -------------------------------------------------------------------------
# [API] 1. 기본 번역 조회 (GET) - 원본 텍스트 번역
# -------------------------------------------------------------------------
@translation_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/translation', methods=['GET'])
def get_target_translation(rubbing_id, target_id):
    try:
        # 위 헬퍼 함수를 사용해 정확한 줄의 텍스트를 가져옴 (치환 없음)
        line_text, char_index = find_target_line_and_replace(rubbing_id, target_id, replacement_char=None)
        
        if isinstance(char_index, str):  # error 메시지인 경우
            return jsonify({'error': char_index}), 404
            
        logger.info(f"[TRANSLATION] 행 번역 요청: {line_text}")

        # 번역 실행
        engine = get_translation_engine()
        result = engine.translate(line_text)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'original': line_text,
                'translation': result.get('translation', ''),
                'reading': result.get('reading', ''),
                'entities': result.get('entities', ''),
                'selected_char_index': char_index  # 선택된 글자의 정확한 인덱스 위치
            }), 200
        else:
            return jsonify({'error': result.get('error', 'Translation failed')}), 500
            
    except Exception as e:
        logger.error(f"[TRANSLATION] 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# -------------------------------------------------------------------------
# [API] 2. 번역 미리보기 (POST) - 선택한 글자 반영 번역
# -------------------------------------------------------------------------
@translation_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/preview-translation', methods=['POST', 'OPTIONS'])
def preview_translation(rubbing_id, target_id):
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        data = request.get_json()
        selected_character = data.get('selected_character')
        
        if not selected_character:
            return jsonify({'error': 'selected_character is required'}), 400
            
        # 헬퍼 함수를 사용해 글자를 치환한 텍스트를 가져옴
        modified_text, char_index = find_target_line_and_replace(rubbing_id, target_id, replacement_char=selected_character)
        
        if isinstance(char_index, str):  # error 메시지인 경우
            return jsonify({'error': char_index}), 404
            
        logger.info(f"[TRANSLATION] 미리보기 요청 (치환: {selected_character}): {modified_text}")
        
        # 번역 실행
        engine = get_translation_engine()
        result = engine.translate(modified_text)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'original': modified_text,  # 치환된 텍스트 반환
                'translation': result.get('translation', ''),
                'reading': result.get('reading', ''),
                'selected_char_index': char_index  # 선택된 글자의 정확한 인덱스 위치
            }), 200
        else:
            return jsonify({'error': result.get('error')}), 500
            
    except Exception as e:
        logger.error(f"[TRANSLATION] 미리보기 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

