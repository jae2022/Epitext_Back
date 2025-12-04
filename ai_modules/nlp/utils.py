"""
유틸리티 함수 모듈
파일 입출력, 텍스트 전처리 등 공통 기능을 제공합니다.
"""

import re
import unicodedata
from typing import Dict, Any


def remove_punctuation(text: str) -> str:
    """
    텍스트에서 구두점과 공백을 제거합니다. [MASK] 토큰은 보존합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        구두점이 제거된 텍스트
    """
    result = []
    i = 0
    
    while i < len(text):
        # [MASK...] 형태의 토큰 보존
        if text[i:i+1] == '[' and 'MASK' in text[i:i+10]:
            end = text.find(']', i)
            if end != -1:
                result.append(text[i:end+1])
                i = end + 1
                continue
        
        # 일반 문자 처리 (구두점과 공백 제외)
        if unicodedata.category(text[i])[0] not in "PZ":
            result.append(text[i])
        i += 1
    
    return "".join(result)


def replace_mask_with_symbol(text: str, symbol: str = "□") -> str:
    """
    [MASK1], [MASK2] 등의 마스크 토큰을 지정된 기호로 치환합니다.
    
    Args:
        text: 원본 텍스트
        symbol: 치환할 기호
        
    Returns:
        마스크가 치환된 텍스트
    """
    return re.sub(r'\[MASK\d+\]', symbol, text)


def normalize_mask_tokens(text: str) -> str:
    """
    [MASK1], [MASK2] 등을 [MASK]로 정규화합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        정규화된 텍스트
    """
    return re.sub(r'\[MASK\d+\]', '[MASK]', text)


def extract_mask_info(json_data: Dict[str, Any]) -> list:
    """
    JSON 데이터에서 마스크 정보를 추출합니다.
    
    Args:
        json_data: 입력 JSON 데이터
        
    Returns:
        마스크 정보 리스트 (order와 type 포함)
    """
    mask_info = []
    for item in json_data.get('results', []):
        if 'MASK' in item.get('type', ''):
            mask_info.append({
                'order': item['order'],
                'type': item['type']
            })
    mask_info.sort(key=lambda x: x['order'])
    return mask_info

