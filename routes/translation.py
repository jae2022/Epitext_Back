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

