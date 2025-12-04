# -*- coding: utf-8 -*-
"""
번역 API 라우트
"""
from flask import Blueprint, request, jsonify
import logging
from ai_modules.translation_engine import get_translation_engine

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


@translation_bp.route('/api/rubbings/<int:rubbing_id>/targets/<int:target_id>/preview-translation', methods=['POST', 'OPTIONS'])
def preview_translation(rubbing_id, target_id):
    """
    선택된 한자로 실시간 번역 미리보기 API
    
    Request Body:
        {
            "selected_character": "선택된 한자"
        }
    
    Response:
        {
            "success": true,
            "translation": "번역 결과"
        }
    """
    try:
        # OPTIONS 요청 처리 (CORS preflight)
        if request.method == 'OPTIONS':
            return jsonify({}), 200
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        selected_character = data.get('selected_character')
        if not selected_character:
            return jsonify({'error': 'selected_character is required'}), 400
        
        logger.info(f"[TRANSLATION-API] 번역 미리보기 요청: rubbing_id={rubbing_id}, target_id={target_id}, character={selected_character}")
        
        # TODO: 실제로는 선택된 한자를 포함한 문맥을 가져와서 번역해야 함
        # 현재는 단순히 선택된 한자만 번역
        engine = get_translation_engine()
        result = engine.translate(selected_character)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'translation': result.get('translation', '')
            }), 200
        else:
            return jsonify({'error': result.get('error', 'Translation failed')}), 500
            
    except Exception as e:
        logger.error(f"[TRANSLATION-API] 번역 미리보기 예외 발생: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

