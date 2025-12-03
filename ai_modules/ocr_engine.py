# -*- coding: utf-8 -*-
"""
================================================================================
OCR Ensemble Module for Epitext AI Project
================================================================================
모듈명: ocr_engine.py (v12.0.0 - Production Ready)
작성일: 2025-12-03
목적: Google Vision API + HRCenterNet 앙상블 기반 한자 OCR 및 손상 영역 탐지
상태: Production Ready
================================================================================
"""
import os
import sys
import io
import cv2
import json
import numpy as np
import torch
import torchvision
import re
import logging
from torch.autograd import Variable
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any

# ================================================================================
# Logging Configuration
# ================================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ================================================================================
# External Model Imports
# ================================================================================
try:
    from ai_modules.models.resnet import ResnetCustom
    from ai_modules.models.HRCenterNet import _HRCenterNet
    logger.info("[INIT] 외부 모델 임포트 완료: ResnetCustom, HRCenterNet")
except ImportError as e:
    logger.error(f"[INIT] 모델 임포트 실패: {e}")
    logger.error("[INIT] 'ai_modules/models' 폴더를 확인하세요.")
    raise

# ================================================================================
# Google Vision API Import
# ================================================================================
try:
    from google.cloud import vision
    HAS_GOOGLE_VISION = True
except ImportError:
    HAS_GOOGLE_VISION = False
    logger.warning("[INIT] google-cloud-vision 패키지가 설치되지 않았습니다. Google OCR 기능이 비활성화됩니다.")

# ================================================================================
# Utility Functions
# ================================================================================
def is_hanja(text: str) -> bool:
    """한자 여부 판별"""
    if not text:
        return False
    return re.match(r'[\u4e00-\u9fff]', text) is not None

def calculate_pixel_density(binary_img: np.ndarray, box: Dict) -> float:
    """바운딩 박스 내 픽셀 밀도 계산"""
    x1, y1 = int(box['min_x']), int(box['min_y'])
    x2, y2 = int(box['max_x']), int(box['max_y'])
    h, w = binary_img.shape
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    roi = binary_img[y1:y2, x1:x2]
    return cv2.countNonZero(roi) / ((x2 - x1) * (y2 - y1))

# ================================================================================
# Configuration Loader
# ================================================================================
def load_ocr_config(config_path: Optional[str] = None) -> Dict:
    """OCR 설정 로드"""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config" / "ocr_config.json")
    
    default_config = {
        "module_info": {
            "name": "ocr_ensemble",
            "version": "12.0.0"
        },
        "model_config": {
            "input_size": 512,
            "output_size": 128,
            "device": "auto"
        },
        "filtering_thresholds": {
            "min_score_hard": 0.30,
            "density_min_hard": 0.10,
            "smart_score_threshold": 0.40,
            "smart_density_threshold": 0.15
        },
        "ink_detection_thresholds": {
            "density_ink_heavy": 0.60,
            "density_ink_partial": 0.38
        },
        "nms_config": {
            "primary_threshold": 0.12,
            "fallback_threshold": 0.08,
            "iou_threshold": 0.05
        },
        "merge_config": {
            "vertical_fragments": {
                "horizontal_center_ratio": 0.5,
                "vertical_gap_ratio": 0.4,
                "aspect_ratio_limit": 1.6
            },
            "google_symbols": {
                "horizontal_center_ratio": 0.5,
                "vertical_gap_ratio": 0.35,
                "aspect_ratio_limit": 1.45
            }
        },
        "ensemble_config": {
            "title_removal_height_ratio": 2.0,
            "top_region_ratio": 0.15,
            "column_grouping_ratio": 0.8,
            "gap_inference_ratio": 1.8,
            "excessive_mask_threshold": 10
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 중첩 딕셔너리 업데이트
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            logger.info(f"[CONFIG] 설정 파일 로드: {config_path}")
        except Exception as e:
            logger.warning(f"[CONFIG] 설정 파일 로드 실패: {e} - 기본값 사용")
    
    return default_config

# ================================================================================
# Text Detection Class
# ================================================================================
class TextDetector:
    """HRCenterNet 기반 텍스트 탐지기"""
    
    def __init__(self, device: torch.device, det_ckpt: str, input_size: int = 512, output_size: int = 128) -> None:
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        self.model = _HRCenterNet(32, 5, 0.1)
        
        if not os.path.exists(det_ckpt):
            raise FileNotFoundError(f"체크포인트 파일 없음: {det_ckpt}")
        
        state = torch.load(det_ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.input_size, self.input_size)),
            torchvision.transforms.ToTensor()
        ])
        
        logger.info(f"[INIT] TextDetector 초기화 완료: {det_ckpt}")
    
    @torch.no_grad()
    def detect(self, image) -> Tuple[List, List]:
        """이미지에서 텍스트 영역 탐지"""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image.convert("RGB")
        
        image_tensor = self.transform(img).unsqueeze_(0)
        inp = Variable(image_tensor).to(self.device, dtype=torch.float)
        
        predict = self.model(inp)
        predict_np = predict.data.cpu().numpy()
        heatmap, offset_y, offset_x, width_map, height_map = predict_np[0]
        
        bbox, score_list = [], []
        Hc, Wc = img.size[1] / self.output_size, img.size[0] / self.output_size
        
        nms_score = 0.12
        idxs = np.where(heatmap.reshape(-1, 1) >= nms_score)[0]
        if len(idxs) == 0:
            nms_score = 0.08
            idxs = np.where(heatmap.reshape(-1, 1) >= nms_score)[0]
        
        for j in idxs:
            row = j // self.output_size
            col = j - row * self.output_size
            bias_x = offset_x[row, col] * Hc
            bias_y = offset_y[row, col] * Wc
            width = width_map[row, col] * self.output_size * Hc
            height = height_map[row, col] * self.output_size * Wc
            score_list.append(float(heatmap[row, col]))
            row = row * Hc + bias_y
            col = col * Wc + bias_x
            top = row - width / 2.0
            left = col - height / 2.0
            bottom = row + width / 2.0
            right = col + height / 2.0
            bbox.append([left, top, max(0.0, right - left), max(0.0, bottom - top)])
        
        if not bbox:
            return [], []
        
        xyxy = [[x, y, x+w, y+h] for x, y, w, h in bbox]
        keep = torchvision.ops.nms(
            torch.tensor(xyxy, dtype=torch.float32),
            scores=torch.tensor(score_list, dtype=torch.float32),
            iou_threshold=0.05
        ).cpu().numpy().tolist()
        
        res_boxes, res_scores = [], []
        W, H = img.size
        for k in keep:
            idx = int(k)
            x, y, w, h = bbox[idx]
            x = max(0.0, min(x, W - 1.0))
            y = max(0.0, min(y, H - 1.0))
            w = max(0.0, min(w, W - x))
            h = max(0.0, min(h, H - y))
            if w > 1 and h > 1:
                res_boxes.append([x, y, w, h])
                res_scores.append(score_list[idx])
        
        logger.info(f"[DETECT] 탐지 완료: {len(res_boxes)}개 박스")
        return res_boxes, res_scores

# ================================================================================
# Vertical Fragment Merging
# ================================================================================
def merge_vertical_fragments(boxes: List[List[float]], scores: List[float]) -> Tuple[List[List[float]], List[float]]:
    """수직으로 분리된 글자 조각 병합"""
    if not boxes:
        return [], []
    
    rects = []
    for i, (b, s) in enumerate(zip(boxes, scores)):
        x, y, w, h = b
        rects.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'x2': x + w, 'y2': y + h,
            'cx': x + w/2, 'cy': y + h/2,
            'score': s
        })
    
    while True:
        rects.sort(key=lambda r: r['y'])
        merged = False
        new_rects = []
        skip_indices = set()
        
        for i in range(len(rects)):
            if i in skip_indices:
                continue
            
            current = rects[i]
            best_cand_idx = -1
            
            for j in range(i + 1, min(i + 5, len(rects))):
                if j in skip_indices:
                    continue
                
                candidate = rects[j]
                avg_w = (current['w'] + candidate['w']) / 2
                dx = abs(current['cx'] - candidate['cx'])
                if dx > avg_w * 0.5:
                    continue
                gap = candidate['y'] - current['y2']
                if gap > avg_w * 0.4:
                    continue
                
                new_h = max(current['y2'], candidate['y2']) - min(current['y'], candidate['y'])
                new_w = max(current['x2'], candidate['x2']) - min(current['x'], candidate['x'])
                new_ratio = new_h / new_w
                cur_ratio = current['h'] / current['w']
                cand_ratio = candidate['h'] / candidate['w']
                is_safe_ratio = new_ratio < 1.6
                is_both_square = (cur_ratio > 0.85 and cand_ratio > 0.85)
                is_overlapped = (gap < -avg_w * 0.2)
                
                if is_safe_ratio and (not is_both_square or is_overlapped):
                    best_cand_idx = j
                    break
            
            if best_cand_idx != -1:
                candidate = rects[best_cand_idx]
                nx = min(current['x'], candidate['x'])
                ny = min(current['y'], candidate['y'])
                nx2 = max(current['x2'], candidate['x2'])
                ny2 = max(current['y2'], candidate['y2'])
                merged_box = {
                    'x': nx, 'y': ny, 'w': nx2 - nx, 'h': ny2 - ny,
                    'x2': nx2, 'y2': ny2,
                    'cx': (nx + nx2)/2, 'cy': (ny + ny2)/2,
                    'score': max(current['score'], candidate['score'])
                }
                new_rects.append(merged_box)
                skip_indices.add(best_cand_idx)
                merged = True
            else:
                new_rects.append(current)
        
        rects = new_rects
        if not merged:
            break
    
    res_boxes = [[r['x'], r['y'], r['w'], r['h']] for r in rects]
    res_scores = [r['score'] for r in rects]
    return res_boxes, res_scores

# ================================================================================
# Google Symbol Merging
# ================================================================================
def merge_google_symbols(symbols: List[Dict]) -> List[Dict]:
    """Google OCR 결과에서 분리된 심볼 병합"""
    if not symbols:
        return []
    
    while True:
        symbols.sort(key=lambda s: s['min_y'])
        merged = False
        new_symbols = []
        skip_indices = set()
        
        for i in range(len(symbols)):
            if i in skip_indices:
                continue
            
            curr = symbols[i]
            best_cand_idx = -1
            
            for j in range(i + 1, min(i + 5, len(symbols))):
                if j in skip_indices:
                    continue
                
                cand = symbols[j]
                avg_w = (curr['width'] + cand['width']) / 2
                dx = abs(curr['center_x'] - cand['center_x'])
                if dx > avg_w * 0.5:
                    continue
                gap = cand['min_y'] - curr['max_y']
                is_touching = gap < (avg_w * 0.35)
                new_h = max(curr['max_y'], cand['max_y']) - min(curr['min_y'], cand['min_y'])
                new_w = max(curr['max_x'], cand['max_x']) - min(curr['min_x'], cand['min_x'])
                ratio = new_h / new_w
                cur_ratio = curr['height'] / curr['width']
                cand_ratio = cand['height'] / cand['width']
                is_both_square = (cur_ratio > 0.85 and cand_ratio > 0.85)
                is_safe_ratio = ratio < 1.45
                is_duplicate_text = (curr['text'] == cand['text'])
                
                if (is_touching and is_safe_ratio and not is_both_square) or is_duplicate_text:
                    best_cand_idx = j
                    break
            
            if best_cand_idx != -1:
                cand = symbols[best_cand_idx]
                merged_sym = {
                    'text': curr['text'],
                    'min_x': min(curr['min_x'], cand['min_x']),
                    'min_y': min(curr['min_y'], cand['min_y']),
                    'max_x': max(curr['max_x'], cand['max_x']),
                    'max_y': max(curr['max_y'], cand['max_y']),
                    'confidence': max(curr['confidence'], cand['confidence']),
                    'source': 'Google'
                }
                merged_sym['width'] = merged_sym['max_x'] - merged_sym['min_x']
                merged_sym['height'] = merged_sym['max_y'] - merged_sym['min_y']
                merged_sym['center_x'] = (merged_sym['min_x'] + merged_sym['max_x']) / 2
                merged_sym['center_y'] = (merged_sym['min_y'] + merged_sym['max_y']) / 2
                new_symbols.append(merged_sym)
                skip_indices.add(best_cand_idx)
                merged = True
            else:
                new_symbols.append(curr)
        
        symbols = new_symbols
        if not merged:
            break
    
    return symbols

# ================================================================================
# Google Vision API OCR
# ================================================================================
def get_google_ocr(content: bytes, google_json_path: Optional[str] = None) -> List[Dict]:
    """Google Vision API를 통한 한자 OCR 수행"""
    if not HAS_GOOGLE_VISION:
        logger.warning("[OCR] Google Vision API 미설치 - Google OCR 건너뜀")
        return []
    
    if google_json_path and os.path.exists(google_json_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_json_path
    
    logger.info("[OCR] Google Vision API 실행 중...")
    
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        context = vision.ImageContext(language_hints=["zh-Hant"])
        response = client.document_text_detection(image=image, image_context=context)
        
        if not response.full_text_annotation:
            logger.warning("[OCR] Google Vision: 텍스트 미검출")
            return []
        
        symbols = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for s in word.symbols:
                            if not is_hanja(s.text):
                                continue
                            
                            v = s.bounding_box.vertices
                            x = [p.x for p in v]
                            y = [p.y for p in v]
                            
                            symbols.append({
                                'text': s.text,
                                'center_x': (min(x) + max(x)) / 2,
                                'center_y': (min(y) + max(y)) / 2,
                                'min_x': min(x), 'max_x': max(x),
                                'min_y': min(y), 'max_y': max(y),
                                'width': max(x) - min(x),
                                'height': max(y) - min(y),
                                'confidence': s.confidence,
                                'source': 'Google'
                            })
        
        original_count = len(symbols)
        symbols = merge_google_symbols(symbols)
        
        if len(symbols) < original_count:
            logger.info(f"[OCR] Google 병합: {original_count} -> {len(symbols)}개")
        
        logger.info(f"[OCR] Google Vision: {len(symbols)}개 인식 완료")
        return symbols
    except Exception as e:
        logger.error(f"[OCR] Google Vision API 오류: {e}")
        return []

# ================================================================================
# Custom Model OCR (HRCenterNet + ResNet)
# ================================================================================
def get_custom_model_ocr(
    image_path: str,
    binary_img: np.ndarray,
    detector: TextDetector,
    recognizer: ResnetCustom,
    config: Dict
) -> List[Dict]:
    """사용자 Custom Model(HRCenterNet + ResNet)을 통한 OCR 수행"""
    logger.info("[OCR] Custom Model (HRCenterNet) 실행 중...")
    
    try:
        pil_img = Image.open(image_path).convert("RGB")
        boxes, scores = detector.detect(pil_img)
        
        if not boxes:
            logger.warning("[OCR] Custom Model: 탐지 결과 없음")
            return []
        
        original_count = len(boxes)
        boxes, scores = merge_vertical_fragments(boxes, scores)
        
        if len(boxes) < original_count:
            logger.info(f"[OCR] Custom 병합: {original_count} -> {len(boxes)}개")
        
        all_heights = [b[3] for b in boxes]
        all_widths = [b[2] for b in boxes]
        median_h = np.median(all_heights) if all_heights else 0
        median_w = np.median(all_widths) if all_widths else 0
        
        crops = []
        for x, y, w, h in boxes:
            crops.append(pil_img.crop((int(x), int(y), int(x+w), int(y+h))))
        
        chars = recognizer(crops) if crops else []
        
        symbols = []
        img_h, img_w = binary_img.shape
        
        thresholds = config.get('filtering_thresholds', {})
        ink_thresholds = config.get('ink_detection_thresholds', {})
        
        MIN_SCORE_HARD = thresholds.get('min_score_hard', 0.30)
        DENSITY_MIN_HARD = thresholds.get('density_min_hard', 0.10)
        SMART_SCORE_THR = thresholds.get('smart_score_threshold', 0.40)
        SMART_DENSITY_THR = thresholds.get('smart_density_threshold', 0.15)
        DENSITY_INK_HEAVY = ink_thresholds.get('density_ink_heavy', 0.60)
        DENSITY_INK_PARTIAL = ink_thresholds.get('density_ink_partial', 0.38)
        
        for char, (x, y, w, h), score in zip(chars, boxes, scores):
            if not char or char == "■":
                continue
            
            box_dict = {'min_x': x, 'min_y': y, 'max_x': x+w, 'max_y': y+h}
            density = calculate_pixel_density(binary_img, box_dict)
            
            if score < MIN_SCORE_HARD or density < DENSITY_MIN_HARD:
                continue
            
            if score < SMART_SCORE_THR and density < SMART_DENSITY_THR:
                continue
            
            is_huge = (h > median_h * 3.5) if median_h > 0 else False
            is_top_title = (y < img_h * 0.15) and (h > median_h * 2.5 or w > median_w * 2.5) if median_h > 0 else False
            
            if median_h > 0 and (is_huge or is_top_title):
                continue
            
            final_text = char
            final_type = 'TEXT'
            
            if density >= DENSITY_INK_HEAVY:
                final_text = '[MASK1]'
                final_type = 'MASK1'
            elif density >= DENSITY_INK_PARTIAL:
                final_text = '[MASK2]'
                final_type = 'MASK2'
            else:
                if not is_hanja(char):
                    continue
            
            symbols.append({
                'text': final_text,
                'type': final_type,
                'center_x': x + w/2,
                'center_y': y + h/2,
                'min_x': x, 'max_x': x + w,
                'min_y': y, 'max_y': y + h,
                'width': w, 'height': h,
                'confidence': float(score),
                'source': 'Custom',
                'density': density
            })
        
        logger.info(f"[OCR] Custom Model: {len(symbols)}개 인식 완료")
        return symbols
    except Exception as e:
        logger.error(f"[OCR] Custom Model 오류: {e}")
        return []

# ================================================================================
# Ensemble Reconstruction (간소화 버전 - 전체 로직은 원본 참조)
# ================================================================================
def ensemble_reconstruction(
    google_syms: List[Dict],
    custom_syms: List[Dict],
    binary_img: np.ndarray,
    config: Dict
) -> Tuple[List[Dict], List[str]]:
    """Google OCR + Custom Model 결과 앙상블 재구성"""
    logger.info("[ENSEMBLE] 앙상블 재구성 시작...")
    
    # 간소화된 앙상블 로직 (전체 로직은 원본 ocr_ensemble.py 참조)
    # 여기서는 기본적인 통합만 수행
    
    all_syms = google_syms + custom_syms
    
    # 간단한 중복 제거
    final_boxes = []
    seen_positions = set()
    
    for sym in all_syms:
        pos_key = (int(sym['center_x']), int(sym['center_y']))
        if pos_key not in seen_positions:
            final_boxes.append(sym)
            seen_positions.add(pos_key)
    
    # 열별 그룹핑 (간소화)
    columns = []
    if final_boxes:
        sorted_x = sorted(final_boxes, key=lambda s: -s['center_x'])
        for s in sorted_x:
            found = False
            for col in columns:
                col_x = sum(c['center_x'] for c in col) / len(col)
                if abs(s['center_x'] - col_x) < s['width'] * 0.8:
                    col.append(s)
                    found = True
                    break
            if not found:
                columns.append([s])
    
    result_lines = []
    for col in columns:
        col.sort(key=lambda s: s['center_y'])
        col_text = [s['text'] for s in col]
        result_lines.append("".join(col_text))
    
    logger.info(f"[ENSEMBLE] 앙상블 완료: {len(final_boxes)}개 박스, {len(result_lines)}개 열")
    return final_boxes, result_lines

# ================================================================================
# OCREngine Class
# ================================================================================
class OCREngine:
    """통합 OCR 엔진 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 1. 설정 로드
        self.config = load_ocr_config(config_path)
        
        # 2. 환경 변수에서 경로 로드
        base_path = os.getenv('OCR_WEIGHTS_BASE_PATH', '/Users/jincerity/Desktop/ocr_weight')
        det_model = os.getenv('OCR_DETECTION_MODEL', 'best.pth')
        rec_model = os.getenv('OCR_RECOGNITION_MODEL', 'best_5000.pt')
        google_json = os.getenv('GOOGLE_CREDENTIALS_JSON', 'tidy-node-479900-m7-a4e08301ce8e.json')
        
        self.det_ckpt = os.path.join(base_path, det_model)
        self.rec_ckpt = os.path.join(base_path, rec_model)
        self.google_json = os.path.join(base_path, google_json)
        
        # Google 자격 증명 설정
        if os.path.exists(self.google_json):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_json
            logger.info(f"[INIT] Google Credential 설정: {self.google_json}")
        else:
            logger.warning(f"[INIT] Google JSON 파일을 찾을 수 없음: {self.google_json}")
        
        # 3. 디바이스 설정
        model_config = self.config.get('model_config', {})
        if model_config.get('device') == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(model_config.get('device', 'cpu'))
        
        logger.info(f"[INIT] OCR Engine Device: {self.device}")
        
        # 4. 모델 지연 로딩
        self.detector = None
        self.recognizer = None
    
    def _load_models(self):
        """모델이 필요할 때 로드"""
        if self.detector is None:
            logger.info(f"[LOAD] Detection Model 로드 중... ({self.det_ckpt})")
            input_size = self.config.get('model_config', {}).get('input_size', 512)
            output_size = self.config.get('model_config', {}).get('output_size', 128)
            self.detector = TextDetector(
                self.device,
                self.det_ckpt,
                input_size,
                output_size
            )
        
        if self.recognizer is None:
            logger.info(f"[LOAD] Recognition Model 로드 중... ({self.rec_ckpt})")
            self.recognizer = ResnetCustom(weight_fn=self.rec_ckpt)
    
    def run_ocr(self, image_path: str) -> Dict:
        """OCR 전체 파이프라인 실행"""
        try:
            self._load_models()
            
            # 1. 이미지 로드 및 이진화
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"이미지를 찾을 수 없음: {image_path}")
            
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.medianBlur(img_gray, 3)
            _, img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
            
            # 2. Google Vision
            with io.open(image_path, 'rb') as f:
                content = f.read()
            google_syms = get_google_ocr(content, self.google_json)
            
            # 3. Custom Model
            custom_syms = get_custom_model_ocr(
                image_path,
                img_binary,
                self.detector,
                self.recognizer,
                self.config
            )
            
            # 4. 앙상블
            final_boxes, result_lines = ensemble_reconstruction(
                google_syms,
                custom_syms,
                img_binary,
                self.config
            )
            
            return {
                "success": True,
                "google_count": len(google_syms),
                "custom_count": len(custom_syms),
                "final_count": len(final_boxes),
                "columns": len(result_lines),
                "results": final_boxes,
                "text_lines": result_lines
            }
        except Exception as e:
            logger.error(f"[OCR] 실행 중 오류: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

# ================================================================================
# Convenience Functions
# ================================================================================
_engine = None

def get_ocr_engine(config_path: Optional[str] = None) -> OCREngine:
    """전역 OCR 엔진 인스턴스 반환"""
    global _engine
    if _engine is None:
        _engine = OCREngine(config_path)
    return _engine

def ocr_and_detect(
    image_path: str,
    config_path: Optional[str] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    device: str = "cuda"
) -> Dict:
    """편의 함수: OCR 실행"""
    engine = get_ocr_engine(config_path)
    return engine.run_ocr(image_path)

