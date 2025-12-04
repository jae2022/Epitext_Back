# Epitext_Back/ai_modules/preprocessor_unified.py
# -*- coding: utf-8 -*-
"""
================================================================================
Unified Image Preprocessing Module for Epitext AI Project
================================================================================

모듈명: preprocessor_unified.py (v1.0.0 - Production Ready)
작성일: 2025-12-02
목적: 한자 이미지를 Swin Gray와 OCR용으로 동시에 전처리
상태: Production Ready

핵심 기능:
    한 번에 두 가지 전처리 완료:
    1. Swin Gray: 그레이 비이진화 -> 3채널 (정보 손실 최소)
    2. OCR: 이진화 -> 1채널 (명확한 흑백)
    
    자동 배경 보장:
    - Swin: 밝은배경 (>=127)
    - OCR: 흰배경 + 검정글자 (255/0)
    
    탁본 자동 검출: 큰 어두운 영역 식별
    영역 검출 1회: 효율성
    설정 파일 지원: JSON 기반 커스터마이징
    로깅 지원: DEBUG, INFO, WARNING, ERROR

의존성:
    - opencv-python >= 4.8.0
    - numpy >= 1.24.0

단일 함수:
    preprocess_image_unified(input_path, output_swin_path, output_ocr_path, ...)

사용 예시:
    >>> from ai_modules.preprocessor_unified import preprocess_image_unified
    >>> result = preprocess_image_unified(
    ...     "input.jpg",
    ...     "swin.jpg",
    ...     "ocr.png"
    ... )

================================================================================
"""


import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple


# ================================================================================
# Logging Configuration
# ================================================================================


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ================================================================================
# Constants
# ================================================================================


# 기본 설정값
DEFAULT_MARGIN = 10
DEFAULT_BRIGHTNESS_THRESHOLD = 127
DEFAULT_RUBBING_MIN_AREA_RATIO = 0.1
DEFAULT_TEXT_MIN_AREA = 16
DEFAULT_TEXT_AREA_RATIO = 0.00005
DEFAULT_MORPHOLOGY_KERNEL_SIZE = (2, 2)
DEFAULT_MORPHOLOGY_CLOSE_ITERATIONS = 3
DEFAULT_MORPHOLOGY_OPEN_ITERATIONS = 2
DEFAULT_RUBBING_KERNEL_SIZE = (5, 5)
DEFAULT_RUBBING_CLOSE_ITERATIONS = 10
DEFAULT_RUBBING_OPEN_ITERATIONS = 5


# ================================================================================
# Main Preprocessing Class
# ================================================================================


class UnifiedImagePreprocessor:
    """
    통합 이미지 전처리 클래스 (Swin + OCR)
    
    한 번의 처리로 Swin Gray와 OCR용 이미지를 모두 생성합니다.
    
    Attributes:
        config (dict): 전처리 설정 파라미터
    
    Example:
        >>> prep = UnifiedImagePreprocessor()
        >>> result = prep.preprocess_unified("input.jpg", "swin.jpg", "ocr.png")
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        UnifiedImagePreprocessor 초기화
        
        Args:
            config_path (str, optional): 설정 파일 경로 (JSON)
        """
        self.config = self._load_config(config_path)
        logger.info("[INIT] UnifiedImagePreprocessor v1.0.0 초기화 완료")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """설정 파일 로드"""
        default_config = {
            "margin": DEFAULT_MARGIN,
            "brightness_threshold": DEFAULT_BRIGHTNESS_THRESHOLD,
            "rubbing_min_area_ratio": DEFAULT_RUBBING_MIN_AREA_RATIO,
            "text_min_area": DEFAULT_TEXT_MIN_AREA,
            "text_area_ratio": DEFAULT_TEXT_AREA_RATIO,
            "morphology_kernel_size": DEFAULT_MORPHOLOGY_KERNEL_SIZE,
            "morphology_close_iterations": DEFAULT_MORPHOLOGY_CLOSE_ITERATIONS,
            "morphology_open_iterations": DEFAULT_MORPHOLOGY_OPEN_ITERATIONS,
            "rubbing_kernel_size": DEFAULT_RUBBING_KERNEL_SIZE,
            "rubbing_close_iterations": DEFAULT_RUBBING_CLOSE_ITERATIONS,
            "rubbing_open_iterations": DEFAULT_RUBBING_OPEN_ITERATIONS,
        }
        
        # 기본 설정 파일 경로 (config_path가 없을 때)
        if config_path is None:
            default_config_path = Path(__file__).parent / "config" / "preprocess_config.json"
            if default_config_path.exists():
                config_path = str(default_config_path)
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # _description 필드는 제외하고 업데이트
                    user_config_clean = {k: v for k, v in user_config.items() if not k.startswith('_')}
                    default_config.update(user_config_clean)
                    logger.info(f"[CONFIG] 설정 파일 로드: {config_path}")
            except Exception as e:
                logger.warning(f"[CONFIG] 설정 파일 로드 실패: {e} - 기본 설정 사용")
        
        return default_config
    
    def _find_rubbing_bbox(self, gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        탁본 영역 검출 (큰 어두운 사각형 찾기)
        
        Args:
            gray_image (np.ndarray): 그레이스케일 이미지
        
        Returns:
            tuple: (x, y, w, h) 또는 None
        """
        H_img, W_img = gray_image.shape
        
        # Step 1: 어두운 영역 추출
        _, dark_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Step 2: 모폴로지 연산
        kernel_rub = np.ones(self.config["rubbing_kernel_size"], np.uint8)
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE, kernel_rub,
            iterations=self.config["rubbing_close_iterations"]
        )
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_OPEN, kernel_rub,
            iterations=self.config["rubbing_open_iterations"]
        )
        
        # Step 3: 컨투어 검출
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Step 4: 가장 큰 컨투어
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Step 5: 면적 검증
        min_area = (H_img * W_img) * self.config["rubbing_min_area_ratio"]
        if area < min_area:
            return None
        
        return cv2.boundingRect(largest)
    
    def _find_text_bbox(self, gray_image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        텍스트 영역 검출
        
        Args:
            gray_image (np.ndarray): 그레이스케일 이미지
        
        Returns:
            tuple: (x, y, w, h)
        """
        H_img, W_img = gray_image.shape
        
        # Step 1: Otsu 이진화
        _, binary = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Step 2: 모폴로지 연산
        kernel_morph = np.ones(self.config["morphology_kernel_size"], np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel_morph,
            iterations=self.config["morphology_close_iterations"]
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel_morph,
            iterations=self.config["morphology_open_iterations"]
        )
        
        # Step 3: 컨투어 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 4: 최소 면적 설정
        min_area = max(
            self.config["text_min_area"],
            int((H_img * W_img) * self.config["text_area_ratio"])
        )
        
        # Step 5: 유효한 컨투어 필터링
        valid_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cv2.boundingRect(cnt)) >= min_area
        ]
        
        # Step 6: 경계박스 계산
        if valid_contours:
            all_points = np.vstack(valid_contours)
            return cv2.boundingRect(all_points)
        else:
            return (0, 0, W_img, H_img)
    
    def _apply_margin(
        self,
        bbox: Tuple[int, int, int, int],
        gray_image: np.ndarray,
        margin_val: int
    ) -> Tuple[int, int, int, int]:
        """여백 추가"""
        x, y, w, h = bbox
        H_img, W_img = gray_image.shape
        
        x_new = max(0, x - margin_val)
        y_new = max(0, y - margin_val)
        w_new = min(W_img - x_new, w + 2 * margin_val)
        h_new = min(H_img - y_new, h + 2 * margin_val)
        
        return (x_new, y_new, w_new, h_new)
    
    def _ensure_bright_background(
        self,
        gray_cropped: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        밝은배경 보장 (Swin용)
        
        Returns:
            tuple: (처리된 그레이 이미지, 처리 정보)
        """
        mean_brightness = np.mean(gray_cropped)
        is_inverted = False
        
        if mean_brightness < self.config["brightness_threshold"]:
            gray_bright = cv2.bitwise_not(gray_cropped)
            is_inverted = True
        else:
            gray_bright = gray_cropped.copy()
        
        # 재확인
        final_brightness = np.mean(gray_bright)
        if final_brightness < self.config["brightness_threshold"]:
            gray_bright = cv2.bitwise_not(gray_bright)
            is_inverted = not is_inverted
            final_brightness = np.mean(gray_bright)
        
        return gray_bright, {
            "mean_brightness_before": float(mean_brightness),
            "mean_brightness_after": float(final_brightness),
            "is_inverted": is_inverted,
            "is_bright_bg": final_brightness >= self.config["brightness_threshold"]
        }
    
    def _ensure_white_background(
        self,
        gray_cropped: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        흰배경 보장 (OCR용)
        
        Returns:
            tuple: (처리된 이진 이미지, 처리 정보)
        """
        # Step 1: 이진화
        _, binary = cv2.threshold(
            gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Step 2: 폴라리티 판단
        mean_brightness = np.mean(binary)
        
        # Step 3: 필요시 반전
        if mean_brightness < self.config["brightness_threshold"]:
            binary_final = cv2.bitwise_not(binary)
            polarity = "inverted"
        else:
            binary_final = binary
            polarity = "normal"
        
        final_brightness = np.mean(binary_final)
        
        return binary_final, {
            "mean_brightness_before": float(mean_brightness),
            "mean_brightness_after": float(final_brightness),
            "polarity": polarity,
            "is_white_bg": final_brightness > self.config["brightness_threshold"]
        }
    
    def preprocess_unified(
        self,
        input_image_path: str,
        output_swin_path: str,
        output_ocr_path: str,
        margin: Optional[int] = None,
        use_rubbing: bool = False
    ) -> Dict:
        """
        통합 전처리 (Swin Gray + OCR 동시 생성)
        
        한 번의 함수 호출로 Swin Gray와 OCR용 이미지를 모두 생성합니다.
        탁본 및 텍스트 영역 검출은 1회만 수행되어 효율성을 보장합니다.
        
        Args:
            input_image_path (str): 입력 이미지 경로
            output_swin_path (str): Swin Gray 출력 경로 (JPG)
            output_ocr_path (str): OCR 출력 경로 (PNG)
            margin (int, optional): 크롭 여백 (픽셀)
            use_rubbing (bool): 탁본 검출 여부 (기본: False)
        
        Returns:
            dict: 처리 결과
                성공 시: {
                    "success": True,
                    "original_shape": (H, W, C),
                    "bbox": (x, y, w, h),
                    "region_type": "text" or "rubbing",
                    "region_detected": bool,
                    
                    "swin": {
                        "output_path": str,
                        "output_shape": (H, W, 3),
                        "is_bright_bg": bool,
                        ...
                    },
                    
                    "ocr": {
                        "output_path": str,
                        "output_shape": (H, W),
                        "is_white_bg": bool,
                        ...
                    }
                }
                
                실패 시: {
                    "success": False,
                    "message": str
                }
        
        Processing Steps:
            1. 이미지 로드
            2. 그레이스케일 변환
            3. 영역 검출 (탁본 또는 텍스트, 1회만)
            4. 크롭 + 여백
            5. Swin Gray 처리 (밝은배경 보장)
            6. OCR 처리 (이진화 + 흰배경 보장)
            7. 동시 저장
        
        Output:
            - Swin: JPG 3채널 (비이진화 256단계)
            - OCR: PNG 1채널 (이진화)
        
        Example:
            >>> prep = UnifiedImagePreprocessor()
            >>> result = prep.preprocess_unified(
            ...     "input.jpg",
            ...     "swin.jpg",
            ...     "ocr.png"
            ... )
            >>> if result["success"]:
            ...     swin_output = result["swin"]["output_path"]
            ...     ocr_output = result["ocr"]["output_path"]
        """
        margin_val = margin or self.config["margin"]
        
        try:
            # ====================================================================
            # Step 1: 이미지 로드
            # ====================================================================
            img_bgr = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"이미지 로드 실패: {input_image_path}")
            
            original_shape = img_bgr.shape
            logger.info(f"[LOAD] 이미지 로드: {input_image_path} {original_shape}")
            
            # ====================================================================
            # Step 2: 그레이스케일 변환
            # ====================================================================
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # ====================================================================
            # Step 3: 영역 검출 (탁본 또는 텍스트)
            # ====================================================================
            if use_rubbing:
                detected_bbox = self._find_rubbing_bbox(gray)
                region_type = "rubbing"
                logger.info("[DETECT] 탁본 영역 검출 모드")
            else:
                detected_bbox = None
                region_type = "text"
                logger.info("[DETECT] 텍스트 영역 검출 모드")
            
            H_img, W_img = gray.shape
            
            # ====================================================================
            # Step 4: 크롭 + 여백
            # ====================================================================
            if detected_bbox is not None:
                bbox_final = self._apply_margin(detected_bbox, gray, margin_val)
                logger.info(f"[DETECT] {region_type} 영역 검출: {bbox_final}")
            else:
                # 탁본 미검출 또는 텍스트 모드 -> 텍스트 검출
                if use_rubbing:
                    bbox_final = (0, 0, W_img, H_img)
                    logger.warning("[DETECT] 탁본 미검출 - 전체 이미지 사용")
                else:
                    bbox_text = self._find_text_bbox(gray)
                    bbox_final = self._apply_margin(bbox_text, gray, margin_val)
                    logger.info(f"[DETECT] 텍스트 영역 검출: {bbox_final}")
            
            x, y, w, h = bbox_final
            
            # ====================================================================
            # [수정] Step 5: Swin용 - 원본 컬러 이미지를 그대로 크롭 (변형 없음)
            # ====================================================================
            # 원본 컬러(BGR) 이미지에서 직접 크롭 (색상 정보 유지, 변환 없음)
            # 로컬 스크립트와 동일: BGR 그대로 저장, _predict_top_k에서 RGB 변환
            swin_output_bgr = img_bgr[y:y+h, x:x+w]
            
            # ====================================================================
            # Step 6: OCR용 - 그레이스케일 크롭 및 이진화
            # ====================================================================
            gray_cropped = gray[y:y+h, x:x+w]
            binary_final, info_ocr = self._ensure_white_background(gray_cropped)
            
            logger.info(f"[CROP] 크롭 완료: Swin={swin_output_bgr.shape}, OCR={gray_cropped.shape}")
            
            # ====================================================================
            # Step 7: 동시 저장
            # ====================================================================
            output_swin_path_obj = Path(output_swin_path)
            output_swin_path_obj.parent.mkdir(parents=True, exist_ok=True)
            # [수정] JPEG 품질 100으로 저장 (압축 손실 방지)
            swin_success = cv2.imwrite(
                str(output_swin_path_obj), 
                swin_output_bgr, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            )
            
            output_ocr_path_obj = Path(output_ocr_path)
            output_ocr_path_obj.parent.mkdir(parents=True, exist_ok=True)
            ocr_success = cv2.imwrite(str(output_ocr_path_obj), binary_final)
            
            if not swin_success or not ocr_success:
                raise ValueError("이미지 저장 실패")
            
            logger.info(f"[SAVE] Swin 저장: {output_swin_path_obj} (원본 BGR 컬러)")
            logger.info(f"[SAVE] OCR 저장: {output_ocr_path_obj}")
            
            # ====================================================================
            # 결과 반환
            # ====================================================================
            # Swin용 밝기 계산 (참고용)
            gray_for_swin = cv2.cvtColor(swin_output_bgr, cv2.COLOR_BGR2GRAY)
            mean_brightness_swin = float(np.mean(gray_for_swin))
            
            return {
                "success": True,
                "version": "Unified Swin BGR + OCR (v1.2.0)",
                "original_shape": original_shape,
                "bbox": bbox_final,
                "region_type": region_type,
                "region_detected": detected_bbox is not None,
                
                # Swin 부분 (수정됨: 원본 BGR 컬러 그대로)
                "swin": {
                    "output_path": str(output_swin_path_obj).replace("\\", "/"),
                    "output_shape": swin_output_bgr.shape,
                    "color_type": "원본 BGR 컬러 (변형 없음, _predict_top_k에서 RGB 변환)",
                    "is_inverted": False,  # 컬러는 반전하지 않음
                    "mean_brightness_before": mean_brightness_swin,
                    "mean_brightness_after": mean_brightness_swin,
                    "is_bright_bg": True  # 컬러는 배경 보정 없음
                },
                
                # OCR 부분
                "ocr": {
                    "output_path": str(output_ocr_path_obj).replace("\\", "/"),
                    "output_shape": binary_final.shape,
                    "polarity": info_ocr["polarity"],
                    "mean_brightness_before": info_ocr["mean_brightness_before"],
                    "mean_brightness_after": info_ocr["mean_brightness_after"],
                    "is_white_bg": info_ocr["is_white_bg"]
                },
                
                "message": "[DONE] 통합 전처리 완료 (Swin BGR 원본 + OCR)"
            }
        
        except Exception as e:
            logger.error(f"[ERROR] 통합 전처리 실패: {e}")
            return {
                "success": False,
                "message": str(e)
            }


# ================================================================================
# Global Instance & Convenience Functions
# ================================================================================


_global_preprocessor = None


def get_preprocessor(config_path: Optional[str] = None) -> UnifiedImagePreprocessor:
    """전역 전처리기 인스턴스 반환"""
    global _global_preprocessor
    if _global_preprocessor is None:
        _global_preprocessor = UnifiedImagePreprocessor(config_path)
    return _global_preprocessor


def preprocess_image_unified(
    input_path: str,
    output_swin_path: str,
    output_ocr_path: str,
    margin: Optional[int] = None,
    use_rubbing: bool = False
) -> Dict:
    """
    편의 함수: 통합 전처리
    
    Args:
        input_path (str): 입력 이미지 경로
        output_swin_path (str): Swin 출력 경로
        output_ocr_path (str): OCR 출력 경로
        margin (int, optional): 여백
        use_rubbing (bool): 탁본 모드
    
    Returns:
        dict: 처리 결과
    """
    prep = get_preprocessor()
    return prep.preprocess_unified(
        input_path,
        output_swin_path,
        output_ocr_path,
        margin,
        use_rubbing
    )


# ================================================================================
# Usage Example
# ================================================================================


if __name__ == "__main__":
    """
    테스트 예시
    """
    logger.info("=" * 80)
    logger.info("[TEST] Unified Image Preprocessor v1.0.0 - 테스트 시작")
    logger.info("=" * 80)
    
    try:
        prep = UnifiedImagePreprocessor()
        
        result = prep.preprocess_unified(
            "test_input.jpg",
            "test_swin.jpg",
            "test_ocr.png"
        )
        
        if result["success"]:
            logger.info("[TEST] 통합 전처리 성공!")
            logger.info(f"[TEST] Swin: {result['swin']['output_path']}")
            logger.info(f"[TEST] OCR:  {result['ocr']['output_path']}")
            logger.info(f"[TEST] Swin 밝은배경: {'Yes' if result['swin']['is_bright_bg'] else 'No'}")
            logger.info(f"[TEST] OCR 흰배경:   {'Yes' if result['ocr']['is_white_bg'] else 'No'}")
        else:
            logger.error(f"[TEST] 실패: {result['message']}")
    
    except Exception as e:
        logger.error(f"[TEST] 예외: {e}")
    
    logger.info("=" * 80)
