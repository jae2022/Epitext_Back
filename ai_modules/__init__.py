# -*- coding: utf-8 -*-
"""
================================================================================
Epitext AI Unified Preprocessing Module
================================================================================

통합 이미지 전처리 패키지 (Swin Gray + OCR 동시 생성)

한 번의 함수 호출로 두 가지 전처리 완료:

  1️⃣  Swin Gray: 그레이 비이진화 (정보 손실 최소) → JPG 3채널

  2️⃣  OCR: 이진화 (명확한 흑백) → PNG 1채널

버전: 1.0.0
상태: ✅ Production Ready

주요 특징:

  ✅ 효율성: 영역 검출 1회 (두 가지 모두 사용)

  ✅ 배경 보장: Swin (밝음) + OCR (하얀색)

  ✅ 탁본 지원: 자동 검출 옵션

  ✅ 설정 가능: JSON 기반 커스터마이징

"""

from .preprocessor_unified import (
    UnifiedImagePreprocessor,
    get_preprocessor,
    preprocess_image_unified
)
from .ocr_engine import (
    get_ocr_engine,
    OCREngine,
    ocr_and_detect
)
from .nlp_engine import (
    get_nlp_engine,
    NLPEngine,
    process_text_with_nlp
)
from .translation_engine import (
    get_translation_engine,
    TranslationEngine
)
from .swin_engine import (
    get_swin_engine,
    SwinMask2Engine,
    MASK2Parser
)

__version__ = "1.0.0"
__author__ = "Epitext Team"

__all__ = [
    "UnifiedImagePreprocessor",
    "get_preprocessor",
    "preprocess_image_unified",
    "get_ocr_engine",
    "OCREngine",
    "ocr_and_detect",
    "get_nlp_engine",
    "NLPEngine",
    "process_text_with_nlp",
    "get_swin_engine",
    "SwinMask2Engine",
    "MASK2Parser",
    "get_translation_engine",
    "TranslationEngine"
]

