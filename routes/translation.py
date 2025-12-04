# -*- coding: utf-8 -*-
"""
번역 API 라우트
"""
from flask import Blueprint, request, jsonify
import logging
import json
from ai_modules.translation_engine import get_translation_engine
from models import RubbingDetail, RestorationTarget

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
    replacement_char가 있다면 해당 위치의 □를 치환하여 반환합니다.
    """
    # 1. 해당 탁본의 모든 Target을 순서대로 조회 (기준점)
    all_targets = RestorationTarget.query.filter_by(rubbing_id=rubbing_id).order_by(RestorationTarget.id).all()
    
    # 2. 현재 Target이 전체 중에서 몇 번째 □인지 찾기 (Global Index)
    try:
        target_ids = [t.id for t in all_targets]
        target_global_index = target_ids.index(target_id)
    except ValueError:
        return None, "Target not found"

    # 3. 구두점 포함 텍스트 가져오기
    detail = RubbingDetail.query.filter_by(rubbing_id=rubbing_id).first()
    if not detail or not detail.text_content_with_punctuation:
        return None, "Text content not found"
        
    try:
        text_lines = json.loads(detail.text_content_with_punctuation)
    except:
        text_lines = []

    # 4. 줄을 순회하며 몇 번째 줄에 내 Target이 있는지 계산
    current_mask_count = 0
    found_line_index = -1
    mask_index_in_line = -1  # 그 줄 안에서 몇 번째 □인지
    
    for idx, line in enumerate(text_lines):
        line_masks = line.count('□')
        # 내 타겟 순번이 현재 줄 범위 안에 있는지 확인
        if current_mask_count <= target_global_index < (current_mask_count + line_masks):
            found_line_index = idx
            mask_index_in_line = target_global_index - current_mask_count
            break
        current_mask_count += line_masks
        
    if found_line_index == -1:
        return None, "Target line mismatch"

    # 5. 찾은 줄 가져오기
    original_line = text_lines[found_line_index]
    
    # 6. 글자 치환 (replacement_char가 있을 때만)
    final_text = original_line
    if replacement_char:
        chars = list(original_line)
        seen_masks = 0
        for i, char in enumerate(chars):
            if char == '□':
                if seen_masks == mask_index_in_line:
                    chars[i] = replacement_char
                    break
                seen_masks += 1
        final_text = "".join(chars)
        
    return final_text, None


# -------------------------------------------------------------------------
# [API] 1. 기본 번역 조회 (GET) - 원본 텍스트 번역
# -------------------------------------------------------------------------
@translation_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/translation', methods=['GET'])
def get_target_translation(rubbing_id, target_id):
    try:
        # 위 헬퍼 함수를 사용해 정확한 줄의 텍스트를 가져옴 (치환 없음)
        line_text, error = find_target_line_and_replace(rubbing_id, target_id, replacement_char=None)
        
        if error:
            return jsonify({'error': error}), 404
            
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
                'entities': result.get('entities', '')
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
        modified_text, error = find_target_line_and_replace(rubbing_id, target_id, replacement_char=selected_character)
        
        if error:
            return jsonify({'error': error}), 404
            
        logger.info(f"[TRANSLATION] 미리보기 요청 (치환: {selected_character}): {modified_text}")
        
        # 번역 실행
        engine = get_translation_engine()
        result = engine.translate(modified_text)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'original': modified_text,  # 치환된 텍스트 반환
                'translation': result.get('translation', ''),
                'reading': result.get('reading', '')
            }), 200
        else:
            return jsonify({'error': result.get('error')}), 500
            
    except Exception as e:
        logger.error(f"[TRANSLATION] 미리보기 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

