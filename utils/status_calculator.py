"""
상태 계산 유틸리티 함수
"""


def calculate_status(processing_time, damage_level):
    """
    상태 계산 로직

    Args:
        processing_time: AI 모델 처리 시간 (초 단위, None이면 처리 중)
        damage_level: 복원 대상 비율 (%)

    Returns:
        "처리중" | "우수" | "양호" | "미흡"
    """
    # 처리 중인 경우
    if processing_time is None:
        return "처리중"

    # 복원 대상 비율에 따라 상태 결정
    if damage_level is None:
        return "처리중"
    
    damage_level = float(damage_level)
    
    if damage_level < 5:
        return "우수"      # 5% 미만
    elif damage_level < 15:
        return "양호"      # 5% 이상 15% 미만
    else:
        return "미흡"      # 15% 이상


def calculate_damage_level(total_characters, restoration_targets):
    """
    복원 대상 비율 계산

    Args:
        total_characters: 전체 글자 수
        restoration_targets: 복원 대상 글자 수

    Returns:
        복원 대상 비율 (%)
    """
    if total_characters == 0:
        return 0.0

    return (restoration_targets / total_characters) * 100


def calculate_f1_score(stroke_match, context_match):
    """
    F1 Score 계산 (신뢰도)
    
    Args:
        stroke_match: 획 일치도 (None 가능)
        context_match: 문맥 일치도
        
    Returns:
        F1 Score (신뢰도)
    """
    if stroke_match is None:
        # 완전 훼손: 문맥 일치도만 사용
        return float(context_match)
    
    # 부분 훼손: F1 Score = 2 * (precision * recall) / (precision + recall)
    # precision = stroke_match, recall = context_match
    precision = float(stroke_match)
    recall = float(context_match)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

