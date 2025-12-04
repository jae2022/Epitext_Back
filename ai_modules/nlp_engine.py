"""
NLP 통합 엔진
구두점 복원 및 MLM 예측을 통합 관리하는 엔진입니다.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .nlp.punctuation_restorer import PunctuationRestorer
from .nlp.mlm_predictor import MLMPredictor
from .nlp.utils import remove_punctuation, replace_mask_with_symbol

logger = logging.getLogger(__name__)


def load_nlp_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    NLP 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        설정 딕셔너리
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "nlp_config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"NLP 설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class NLPEngine:
    """NLP 처리 통합 엔진 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        NLP 엔진을 초기화합니다.
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        """
        self.config = load_nlp_config(config_path)
        
        # 디바이스 설정
        dev_cfg = self.config.get('device', 'auto')
        if dev_cfg == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = dev_cfg
        
        logger.info(f"[NLP] Device: {self.device}")
        
        # 모델 캐시 경로 (환경 변수 또는 기본값)
        self.base_model_dir = os.getenv(
            'AI_MODEL_DIR', 
            str(Path(__file__).parent.parent / "models")
        )
        
        # 서브 모듈 초기화 (지연 로딩)
        self.punc_restorer = None
        self.mlm_predictor = None
    
    def _load_models(self):
        """필요할 때 모델을 메모리에 로드"""
        if self.punc_restorer is None:
            logger.info("[NLP] 구두점 복원 모델 로드 중...")
            self.punc_restorer = PunctuationRestorer(
                self.config, 
                self.base_model_dir, 
                self.device
            )
            self.punc_restorer.download_model()
            self.punc_restorer.load_model()
            
        if self.mlm_predictor is None:
            logger.info("[NLP] MLM 모델 로드 중...")
            self.mlm_predictor = MLMPredictor(self.config, self.device)
            self.mlm_predictor.load_model()
    
    def process_text(
        self, 
        raw_text: str,
        ocr_results: Optional[List[Dict]] = None,
        add_space: bool = True,
        reduce_punc: bool = True
    ) -> Dict[str, Any]:
        """
        텍스트 처리 파이프라인:
        1. 구두점 제거 (전처리)
        2. 구두점 복원
        3. [MASK] 예측
        
        Args:
            raw_text: 원본 텍스트 (구두점 포함 가능)
            add_space: 구두점 뒤 공백 추가 여부
            reduce_punc: 구두점 단순화 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        self._load_models()
        
        try:
            # 1. 전처리 (구두점 제거, [MASK] 보존)
            clean_text = remove_punctuation(raw_text)
            logger.info(f"[NLP] 구두점 제거 완료: {len(clean_text)} 글자")
            
            # 2. 구두점 복원
            punctuated_text = self.punc_restorer.restore_punctuation(
                clean_text,
                add_space=add_space,
                reduce=reduce_punc
            )
            logger.info(f"[NLP] 구두점 복원 완료: {len(punctuated_text)} 글자")
            
            # 3. MLM 예측
            mask_predictions = self.mlm_predictor.predict_masks(punctuated_text)
            logger.info(f"[NLP] MLM 예측 완료: {len(mask_predictions)}개 마스크")
            
            # 4. 출력용 텍스트 생성 ([MASK] -> □)
            mask_replacement = self.config['tokens']['mask_replacement']
            final_text = replace_mask_with_symbol(
                punctuated_text, 
                mask_replacement
            )
            
            # Extract mask info from OCR results or original text
            mask_info_list = []
            if ocr_results:
                # Use OCR results to get order and type
                for item in ocr_results:
                    if 'MASK' in item.get('type', ''):
                        mask_info_list.append({
                            'order': item.get('order', 0),
                            'type': item.get('type', 'MASK2'),
                            'text': item.get('text', '')
                        })
            else:
                # Fallback: extract from text
                i = 0
                while i < len(raw_text):
                    if raw_text[i] == '[' and 'MASK' in raw_text[i:i+10]:
                        end = raw_text.find(']', i)
                        if end != -1:
                            mask_text = raw_text[i:end+1]
                            mask_type = 'MASK1' if 'MASK1' in mask_text else 'MASK2'
                            mask_info_list.append({
                                'order': len(mask_info_list),  # Sequential order
                                'type': mask_type,
                                'text': mask_text
                            })
                            i = end + 1
                            continue
                    i += 1
            
            # Format results according to specification
            formatted_results = []
            for idx, pred_list in enumerate(mask_predictions):
                if idx < len(mask_info_list):
                    mask_info = mask_info_list[idx]
                    formatted_results.append({
                        "order": mask_info['order'],
                        "type": mask_info['type'],
                        "top_20": pred_list[:20]  # Top-20 predictions
                    })
                else:
                    # Fallback if mask_info_list is shorter
                    formatted_results.append({
                        "order": idx,
                        "type": "MASK2",
                        "top_20": pred_list[:20]
                    })
            
            # Calculate statistics
            top1_probs = [preds[0]['probability'] for preds in mask_predictions if preds]
            statistics = {
                "top1_probability_avg": float(sum(top1_probs) / len(top1_probs)) if top1_probs else 0.0,
                "top1_probability_min": float(min(top1_probs)) if top1_probs else 0.0,
                "top1_probability_max": float(max(top1_probs)) if top1_probs else 0.0,
                "total_masks": len(mask_predictions)
            }
            
            return {
                "punctuated_text_with_masks": final_text,
                "results": formatted_results,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"[NLP] 처리 중 오류: {e}", exc_info=True)
            return {
                "success": False, 
                "error": str(e)
            }
    
    def restore_punctuation_only(
        self,
        text: str,
        add_space: bool = True,
        reduce_punc: bool = True
    ) -> Dict[str, Any]:
        """
        구두점 복원만 수행합니다 (MLM 예측 제외).
        
        Args:
            text: 입력 텍스트
            add_space: 구두점 뒤 공백 추가 여부
            reduce_punc: 구두점 단순화 여부
            
        Returns:
            구두점 복원 결과
        """
        self._load_models()
        
        try:
            clean_text = remove_punctuation(text)
            punctuated_text = self.punc_restorer.restore_punctuation(
                clean_text,
                add_space=add_space,
                reduce=reduce_punc
            )
            
            return {
                "success": True,
                "original_text": text,
                "clean_text": clean_text,
                "punctuated_text": punctuated_text
            }
        except Exception as e:
            logger.error(f"[NLP] 구두점 복원 중 오류: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_masks_only(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        MLM 예측만 수행합니다 (구두점 복원 제외).
        
        Args:
            text: 마스크가 포함된 텍스트
            
        Returns:
            MLM 예측 결과
        """
        self._load_models()
        
        try:
            mask_predictions = self.mlm_predictor.predict_masks(text)
            
            return {
                "success": True,
                "predictions": mask_predictions,
                "mask_count": len(mask_predictions)
            }
        except Exception as e:
            logger.error(f"[NLP] MLM 예측 중 오류: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


# ================================================================================
# Global Accessor
# ================================================================================
_nlp_engine = None


def get_nlp_engine(config_path: Optional[str] = None) -> NLPEngine:
    """
    전역 NLP 엔진 인스턴스를 반환합니다 (싱글톤 패턴).
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        NLPEngine 인스턴스
    """
    global _nlp_engine
    if _nlp_engine is None:
        _nlp_engine = NLPEngine(config_path)
    return _nlp_engine


def process_text_with_nlp(
    text: str,
    ocr_results: Optional[List[Dict]] = None,
    config_path: Optional[str] = None,
    add_space: bool = True,
    reduce_punc: bool = True
) -> Dict[str, Any]:
    """
    편의 함수: 텍스트를 NLP 파이프라인으로 처리합니다.
    
    Args:
        text: 입력 텍스트
        ocr_results: OCR 결과 리스트 (order, type 정보 포함)
        config_path: 설정 파일 경로
        add_space: 구두점 뒤 공백 추가 여부
        reduce_punc: 구두점 단순화 여부
        
    Returns:
        처리 결과 딕셔너리
    """
    engine = get_nlp_engine(config_path)
    return engine.process_text(text, ocr_results=ocr_results, add_space=add_space, reduce_punc=reduce_punc)

