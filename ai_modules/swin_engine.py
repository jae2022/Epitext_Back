# -*- coding: utf-8 -*-
"""
================================================================================
Swin Transformer MASK2 Restoration Engine for Epitext AI Project
================================================================================
모듈명: swin_engine.py (v1.0.0 - Production Ready)
작성일: 2025-12-04
목적: OCR 결과에서 MASK2(부분 오염) 영역을 Swin Transformer로 복원
상태: Production Ready
================================================================================
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

# ================================================================================
# Logging Configuration
# ================================================================================
logger = logging.getLogger(__name__)

# ================================================================================
# Data Classes
# ================================================================================
@dataclass
class BBox:
    """바운딩 박스 데이터 클래스"""
    x: int
    y: int
    w: int
    h: int

@dataclass
class MASK2Item:
    """MASK2 항목 데이터 클래스"""
    order: int
    bbox: BBox
    original_text: str
    original_confidence: float

# ================================================================================
# Utility Functions
# ================================================================================
def load_swin_config(config_path: Optional[str] = None) -> Dict:
    """설정 파일 로드"""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config" / "swin_config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ================================================================================
# MASK2 Parser
# ================================================================================
class MASK2Parser:
    """
    OCR 결과에서 MASK2 항목 추출
    
    입력 형식:
        - Dict: {"results": [...]}
        - List: [...]
    """
    
    @staticmethod
    def parse(ocr_results: Union[List[Dict], Dict]) -> List[MASK2Item]:
        """
        OCR 결과에서 MASK2 항목만 추출
        
        Args:
            ocr_results: OCR 엔진 결과 (Dict 또는 List)
            
        Returns:
            List[MASK2Item]: MASK2 항목 리스트
        """
        # 입력이 Dict인지 List인지 확인
        if isinstance(ocr_results, dict):
            results_list = ocr_results.get('results', [])
        else:
            results_list = ocr_results
        
        mask2_items: List[MASK2Item] = []
        
        for idx, item in enumerate(results_list):
            if not isinstance(item, dict):
                continue
                
            item_type = str(item.get('type', ''))
            
            # MASK2 타입만 처리
            if 'MASK2' not in item_type:
                continue
            
            # Box 좌표 처리
            # [수정] 로컬 스크립트와 동일하게 round() 적용하여 좌표 오차 제거
            x, y, w, h = 0, 0, 0, 0
            
            # ocr_engine.py 출력 스타일: min_x, min_y, max_x, max_y
            if 'min_x' in item and 'min_y' in item:
                # [핵심] 로컬 스크립트와 동일한 반올림(Round) 적용
                x = int(round(float(item['min_x'])))
                y = int(round(float(item['min_y'])))
                # width/height 계산 시에도 round 적용
                w_float = float(item.get('width', float(item.get('max_x', 0)) - float(item['min_x'])))
                h_float = float(item.get('height', float(item.get('max_y', 0)) - float(item['min_y'])))
                w = int(round(w_float))
                h = int(round(h_float))
            # box 리스트 스타일: [x1, y1, x2, y2]
            elif 'box' in item and isinstance(item['box'], list) and len(item['box']) == 4:
                x1, y1, x2, y2 = map(float, item['box'])
                x = int(round(x1))
                y = int(round(y1))
                w = int(round(x2 - x1))
                h = int(round(y2 - y1))
            else:
                continue
            
            # 유효성 검증
            if w > 0 and h > 0:
                bbox = BBox(x=x, y=y, w=w, h=h)
                # order가 없으면 인덱스 사용
                order = item.get('order', idx)
                mask2_items.append(MASK2Item(
                    order=order,
                    bbox=bbox,
                    original_text=item.get('text', ''),
                    original_confidence=float(item.get('confidence', 0.0))
                ))
        
        logger.info(f"[SWIN-PARSE] MASK2 추출 완료: {len(mask2_items)}개")
        return mask2_items

# ================================================================================
# Swin Transformer Engine
# ================================================================================
class SwinMask2Engine:
    """
    MASK2 전용 Swin Transformer 복원 엔진
    
    싱글톤 패턴으로 모델을 한 번만 로드하여 메모리 효율성 향상
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Swin MASK2 엔진 초기화
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        """
        self.config = load_swin_config(config_path)
        
        # 1. 환경 변수에서 체크포인트 경로 로드
        base_path = os.getenv('SWIN_CHECKPOINT_PATH', '')
        model_file = os.getenv('SWIN_MODEL_FILE', 'last_checkpoint.pth')
        
        # [수정] best_model.pth 우선 시도, 없으면 last_checkpoint.pth 또는 환경 변수 지정 파일
        if base_path:
            # best_model.pth 우선 확인
            best_model_path = os.path.join(base_path, 'best_model.pth')
            if os.path.exists(best_model_path):
                self.checkpoint_path = best_model_path
                logger.info(f"[SWIN-INIT] best_model.pth 사용: {self.checkpoint_path}")
            else:
                # 환경 변수로 지정된 파일 또는 last_checkpoint.pth 확인
                specified_path = os.path.join(base_path, model_file)
                if os.path.exists(specified_path):
                    self.checkpoint_path = specified_path
                    logger.info(f"[SWIN-INIT] {model_file} 사용: {self.checkpoint_path}")
                else:
                    logger.warning(f"[SWIN-INIT] 체크포인트 파일 없음: {base_path}")
                    logger.warning(f"  - 시도한 파일: best_model.pth, {model_file}")
                    self.model = None
                    self.transform = None
                    self.num_classes = None
                    self.idx2char = None
                    self.device = None
                    return
        else:
            # 폴백 경로 (로컬 테스트용)
            fallback_best = "/Users/jincerity/Desktop/swin_checkpoint/best_model.pth"
            fallback_last = "/Users/jincerity/Desktop/swin_checkpoint/last_checkpoint.pth"
            
            if os.path.exists(fallback_best):
                self.checkpoint_path = fallback_best
                logger.info(f"[SWIN-INIT] Fallback (best_model.pth) 사용: {self.checkpoint_path}")
            elif os.path.exists(fallback_last):
                self.checkpoint_path = fallback_last
                logger.info(f"[SWIN-INIT] Fallback (last_checkpoint.pth) 사용: {self.checkpoint_path}")
            else:
                logger.warning(f"[SWIN-INIT] 체크포인트 없음. 엔진을 사용할 수 없습니다.")
                logger.warning(f"  - 환경 변수 SWIN_CHECKPOINT_PATH가 설정되지 않았습니다.")
                logger.warning(f"  - 폴백 경로도 존재하지 않습니다: {fallback_best}, {fallback_last}")
                self.model = None
                self.transform = None
                self.num_classes = None
                self.idx2char = None
                self.device = None
                return
        
        # 2. 디바이스 설정
        dev_cfg = self.config['model_config'].get('device', 'auto')
        if dev_cfg == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_cfg)
        
        # 3. 모델 로드 (제공된 코드 기준으로 __init__에서 바로 로드)
        self.model = None
        self.transform = None
        self.num_classes = None
        self.idx2char = None
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        if self.model is not None:
            return
        
        try:
            logger.info(f"[SWIN-INIT] 모델 로딩 시작... ({self.device})")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            self.num_classes = checkpoint["num_classes"]
            self.idx2char = {int(k): v for k, v in checkpoint["char_mapping"].get("idx2char", {}).items()}
            
            model_name = self.config['model_config']['model_name']
            img_size = self.config['model_config']['img_size']
            
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=self.num_classes,
                img_size=img_size
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            # Transform 설정
            mean = self.config['preprocessing']['normalize_mean']
            std = self.config['preprocessing']['normalize_std']
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            logger.info("[SWIN-INIT] 모델 준비 완료")
            
        except Exception as e:
            logger.error(f"[SWIN-INIT] 모델 로드 중 오류 발생: {e}")
            self.model = None
    
    def run_swin_restoration(
        self, 
        image_path: str, 
        ocr_results: Union[List[Dict], Dict]
    ) -> Dict:
        """
        통합 복원 실행 함수
        
        Args:
            image_path: 전처리된 이미지 경로 (swin_path 권장)
            ocr_results: OCR 엔진의 실행 결과 (Dict 또는 List)
            
        Returns:
            Dict: 복원 결과가 포함된 딕셔너리
                {
                    "success": True,
                    "restored_count": 10,
                    "results": [
                        {
                            "order": 36,
                            "type": "MASK2",
                            "original_text": "[MASK2]",
                            "candidates": [
                                {"character": "䟽", "confidence": 0.381},
                                ...
                            ]
                        },
                        ...
                    ],
                    "statistics": {
                        "top1_probability_avg": 0.45,
                        "top1_probability_min": 0.12,
                        "top1_probability_max": 0.89
                    }
                }
        """
        if self.model is None:
            logger.error("[SWIN] 모델이 로드되지 않았습니다.")
            return {
                "results": [],
                "statistics": {
                    "total_masks": 0,
                    "top1_probability_avg": 0.0,
                    "top1_probability_min": 0.0,
                    "top1_probability_max": 0.0
                }
            }
        
        try:
            # 1. 이미지 로드 (BGR)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image not found: {image_path}")
            
            logger.info(f"[SWIN] 이미지 로드 완료: {image.shape}")
            
            # 2. MASK2 파싱 (Round 적용됨)
            mask2_items = MASK2Parser.parse(ocr_results)
            
            # [디버깅] 첫 번째 MASK2 좌표 확인
            if mask2_items:
                first_item = mask2_items[0]
                logger.info(f"[SWIN-DEBUG] 첫 번째 MASK2 (order={first_item.order}): "
                           f"bbox=[{first_item.bbox.x}, {first_item.bbox.y}, "
                           f"{first_item.bbox.w}, {first_item.bbox.h}], "
                           f"image_shape={image.shape}")
            
            if not mask2_items:
                logger.info("[SWIN] 복원할 MASK2 항목이 없습니다.")
                return {
                    "results": [],
                    "statistics": {
                        "total_masks": 0,
                        "top1_probability_avg": 0.0,
                        "top1_probability_min": 0.0,
                        "top1_probability_max": 0.0
                    }
                }
            
            # 3. 배치 복원
            restored_results = self._restore_batch(image, mask2_items)
            
            # 4. 통계 계산
            statistics = self._calculate_statistics(restored_results)
            
            # 요청한 형식: results와 statistics만 반환
            return {
                "results": restored_results,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"[SWIN] 실행 중 오류: {e}", exc_info=True)
            return {
                "results": [],
                "statistics": {
                    "total_masks": 0,
                    "top1_probability_avg": 0.0,
                    "top1_probability_min": 0.0,
                    "top1_probability_max": 0.0
                }
            }
    
    def _restore_batch(
        self, 
        image: np.ndarray, 
        mask2_items: List[MASK2Item]
    ) -> List[Dict]:
        """배치 복원 수행"""
        results = []
        top_k = self.config['model_config']['top_k']
        total = len(mask2_items)
        
        logger.info(f"[SWIN-RESTORE] {total}개 MASK2 복원 시작")
        
        for idx, item in enumerate(mask2_items, 1):
            # 이미지 크롭
            # [디버깅] order=5일 때만 상세 로깅
            if item.order == 5:
                logger.info(f"[SWIN-DEBUG] order=5 크롭 전: "
                           f"bbox=[{item.bbox.x}, {item.bbox.y}, {item.bbox.w}, {item.bbox.h}], "
                           f"image_shape={image.shape}")
            
            cropped = self._crop_image(image, item.bbox)
            
            if cropped is None:
                logger.warning(f"[SWIN-RESTORE] [{idx}/{total}] 크롭 실패: order={item.order}")
                results.append({
                    "order": item.order,
                    "type": "MASK2",
                    "top_10": []
                })
                continue
            
            # [디버깅] order=5일 때만 크롭 결과 로깅
            if item.order == 5:
                logger.info(f"[SWIN-DEBUG] order=5 크롭 후: shape={cropped.shape}")
            
            # Swin 추론 (이미 token/probability 형식으로 반환됨)
            predictions = self._predict_top_k(cropped, k=top_k)
            
            # 결과 추가 (변환 불필요, 직접 사용)
            results.append({
                "order": item.order,
                "type": "MASK2",
                "top_10": predictions  # 이미 token/probability 형식
            })
            
            # 진행 상황 로깅
            if predictions:
                top1 = predictions[0]
                logger.info(
                    f"[SWIN-RESTORE] [{idx}/{total}] order={item.order} "
                    f"-> '{top1['token']}' ({top1['probability']:.2%})"
                )
            else:
                logger.info(f"[SWIN-RESTORE] [{idx}/{total}] order={item.order} -> 예측 실패")
        
        logger.info(f"[SWIN-RESTORE] 배치 복원 완료: {len(results)}개")
        return results
    
    def _crop_image(
        self, 
        image: np.ndarray, 
        bbox: BBox
    ) -> Optional[np.ndarray]:
        """
        바운딩 박스 기반 이미지 크롭
        
        [수정] 로컬 스크립트와 완전히 동일한 로직 적용
        """
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        img_h, img_w = image.shape[:2]
        
        # 경계 검증
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        
        # 경계 클리핑
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y
        
        if w <= 0 or h <= 0:
            return None
        
        try:
            cropped = image[y:y+h, x:x+w]
            
            # 최소 크기 검증
            if cropped.shape[0] < 5 or cropped.shape[1] < 5:
                return None
            
            return cropped
        except Exception:
            # 로컬 스크립트와 동일: 예외 메시지 없이 None 반환
            return None
    
    def _predict_top_k(
        self, 
        cropped: np.ndarray, 
        k: int = 10
    ) -> List[Dict]:
        """
        Top-K 예측 수행
        
        [수정] 로컬 스크립트와 동일한 형식으로 반환 (token, probability)
        
        Returns:
            List[Dict]: [{"token": "䟽", "probability": 0.381}, ...]
        """
        try:
            # BGR -> RGB -> PIL
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_rgb)
            
            # Transform 적용
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model(tensor)  # (1, num_classes)
                probs = torch.softmax(outputs, dim=1)[0]  # (num_classes,)
            
            # Top-K 추출
            top_probs, top_indices = torch.topk(probs, k=min(k, len(probs)))
            
            # 결과 포맷팅 (로컬 스크립트와 동일: token, probability)
            results = []
            for prob, idx in zip(top_probs, top_indices):
                char = self.idx2char.get(int(idx), "?")
                results.append({
                    "token": char,  # character 대신 token
                    "probability": float(prob)  # confidence 대신 probability
                })
            
            return results
            
        except Exception as e:
            logger.error(f"[SWIN-PREDICT] 예측 오류: {e}")
            return []
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """통계 계산"""
        valid_results = [r for r in results if r.get('top_10')]
        
        if not valid_results:
            return {
                "top1_probability_avg": 0.0,
                "top1_probability_min": 0.0,
                "top1_probability_max": 0.0,
                "total_masks": len(results)
            }
        
        top1_probs = [
            r['top_10'][0]['probability'] 
            for r in valid_results 
            if r.get('top_10')
        ]
        
        return {
            "top1_probability_avg": float(np.mean(top1_probs)) if top1_probs else 0.0,
            "top1_probability_min": float(np.min(top1_probs)) if top1_probs else 0.0,
            "top1_probability_max": float(np.max(top1_probs)) if top1_probs else 0.0,
            "total_masks": len(results)
        }

# ================================================================================
# Global Accessor
# ================================================================================
_swin_engine = None

def get_swin_engine(config_path: Optional[str] = None) -> SwinMask2Engine:
    """
    Swin 엔진 싱글톤 인스턴스 반환
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        SwinMask2Engine: Swin 엔진 인스턴스
    """
    global _swin_engine
    if _swin_engine is None:
        _swin_engine = SwinMask2Engine(config_path)
    return _swin_engine

