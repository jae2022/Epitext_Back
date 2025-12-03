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
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

# ================================================================================
# Logging Configuration
# ================================================================================
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
    raise

# ================================================================================
# Google Vision API Import
# ================================================================================
try:
    from google.cloud import vision
    HAS_GOOGLE_VISION = True
except ImportError:
    HAS_GOOGLE_VISION = False
    logger.warning("[INIT] google-cloud-vision 패키지가 설치되지 않았습니다.")

# ================================================================================
# Utility Functions
# ================================================================================
def is_hanja(text: str) -> bool:
    if not text: return False
    return re.match(r'[\u4e00-\u9fff]', text) is not None

def calculate_pixel_density(binary_img: np.ndarray, box: Dict) -> float:
    x1, y1 = int(box['min_x']), int(box['min_y'])
    x2, y2 = int(box['max_x']), int(box['max_y'])
    h, w = binary_img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return 0.0
    roi = binary_img[y1:y2, x1:x2]
    return cv2.countNonZero(roi) / ((x2 - x1) * (y2 - y1))

def load_ocr_config(config_path: Optional[str] = None) -> Dict:
    """설정 파일 로드"""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config" / "ocr_config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ================================================================================
# Text Detection Class
# ================================================================================
class TextDetector:
    def __init__(self, device: torch.device, det_ckpt: str, config: Dict):
        self.device = device
        self.config = config
        self.input_size = config['model_config']['input_size']
        self.output_size = config['model_config']['output_size']
        
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

    @torch.no_grad()
    def detect(self, image) -> Tuple[List, List]:
        if isinstance(image, str): img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray): img = Image.fromarray(image).convert("RGB")
        else: img = image.convert("RGB")
        
        image_tensor = self.transform(img).unsqueeze_(0)
        inp = Variable(image_tensor).to(self.device, dtype=torch.float)
        
        predict = self.model(inp)
        predict_np = predict.data.cpu().numpy()
        heatmap, offset_y, offset_x, width_map, height_map = predict_np[0]
        
        bbox, score_list = [], []
        Hc, Wc = img.size[1] / self.output_size, img.size[0] / self.output_size
        
        # Config에서 NMS 임계값 로드
        nms_cfg = self.config.get('nms_config', {})
        nms_score = nms_cfg.get('primary_threshold', 0.12)
        
        idxs = np.where(heatmap.reshape(-1, 1) >= nms_score)[0]
        if len(idxs) == 0:
            nms_score = nms_cfg.get('fallback_threshold', 0.08)
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
            
        if not bbox: return [], []
        
        xyxy = [[x, y, x+w, y+h] for x, y, w, h in bbox]
        keep = torchvision.ops.nms(
            torch.tensor(xyxy, dtype=torch.float32),
            scores=torch.tensor(score_list, dtype=torch.float32),
            iou_threshold=nms_cfg.get('iou_threshold', 0.05)
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
                
        return res_boxes, res_scores

# ================================================================================
# Merging Logics (Config 적용)
# ================================================================================
def merge_vertical_fragments(boxes, scores, config):
    if not boxes: return [], []
    rects = [{'x': b[0], 'y': b[1], 'w': b[2], 'h': b[3], 
              'x2': b[0]+b[2], 'y2': b[1]+b[3], 
              'cx': b[0]+b[2]/2, 'cy': b[1]+b[3]/2, 'score': s} 
             for b, s in zip(boxes, scores)]
    
    cfg = config['merge_config']['vertical_fragments']
    
    while True:
        rects.sort(key=lambda r: r['y'])
        merged = False
        new_rects, skip_indices = [], set()
        
        for i in range(len(rects)):
            if i in skip_indices: continue
            current = rects[i]
            best_cand_idx = -1
            
            for j in range(i + 1, min(i + 5, len(rects))):
                if j in skip_indices: continue
                candidate = rects[j]
                
                avg_w = (current['w'] + candidate['w']) / 2
                if abs(current['cx'] - candidate['cx']) > avg_w * cfg['horizontal_center_ratio']: continue
                if (candidate['y'] - current['y2']) > avg_w * cfg['vertical_gap_ratio']: continue
                
                new_h = max(current['y2'], candidate['y2']) - min(current['y'], candidate['y'])
                new_w = max(current['x2'], candidate['x2']) - min(current['x'], candidate['x'])
                
                is_safe_ratio = (new_h / new_w) < cfg['aspect_ratio_limit']
                cur_square = (current['h'] / current['w']) > 0.85
                cand_square = (candidate['h'] / candidate['w']) > 0.85
                is_overlapped = (candidate['y'] - current['y2']) < -avg_w * 0.2
                
                if is_safe_ratio and (not (cur_square and cand_square) or is_overlapped):
                    best_cand_idx = j
                    break
            
            if best_cand_idx != -1:
                cand = rects[best_cand_idx]
                nx, ny = min(current['x'], cand['x']), min(current['y'], cand['y'])
                nx2, ny2 = max(current['x2'], cand['x2']), max(current['y2'], cand['y2'])
                new_rects.append({
                    'x': nx, 'y': ny, 'w': nx2-nx, 'h': ny2-ny,
                    'x2': nx2, 'y2': ny2, 'cx': (nx+nx2)/2, 'cy': (ny+ny2)/2,
                    'score': max(current['score'], cand['score'])
                })
                skip_indices.add(best_cand_idx)
                merged = True
            else:
                new_rects.append(current)
        rects = new_rects
        if not merged: break
        
    return [[r['x'], r['y'], r['w'], r['h']] for r in rects], [r['score'] for r in rects]

def merge_google_symbols(symbols, config):
    if not symbols: return []
    cfg = config['merge_config']['google_symbols']
    
    while True:
        symbols.sort(key=lambda s: s['min_y'])
        merged = False
        new_symbols, skip_indices = [], set()
        
        for i in range(len(symbols)):
            if i in skip_indices: continue
            curr = symbols[i]
            best_cand_idx = -1
            
            for j in range(i + 1, min(i + 5, len(symbols))):
                if j in skip_indices: continue
                cand = symbols[j]
                
                avg_w = (curr['width'] + cand['width']) / 2
                if abs(curr['center_x'] - cand['center_x']) > avg_w * cfg['horizontal_center_ratio']: continue
                
                gap = cand['min_y'] - curr['max_y']
                is_touching = gap < (avg_w * cfg['vertical_gap_ratio'])
                
                new_h = max(curr['max_y'], cand['max_y']) - min(curr['min_y'], cand['min_y'])
                new_w = max(curr['max_x'], cand['max_x']) - min(curr['min_x'], cand['min_x'])
                
                is_both_square = (curr['height']/curr['width'] > 0.85) and (cand['height']/cand['width'] > 0.85)
                is_safe_ratio = (new_h / new_w) < cfg['aspect_ratio_limit']
                is_duplicate = (curr['text'] == cand['text'])
                
                if (is_touching and is_safe_ratio and not is_both_square) or is_duplicate:
                    best_cand_idx = j
                    break
            
            if best_cand_idx != -1:
                cand = symbols[best_cand_idx]
                merged_sym = {
                    'text': curr['text'],
                    'min_x': min(curr['min_x'], cand['min_x']), 'min_y': min(curr['min_y'], cand['min_y']),
                    'max_x': max(curr['max_x'], cand['max_x']), 'max_y': max(curr['max_y'], cand['max_y']),
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
        if not merged: break
    return symbols

# ================================================================================
# Models Execution
# ================================================================================
def get_google_ocr(content: bytes, config: Dict, google_json_path: Optional[str] = None) -> List[Dict]:
    if not HAS_GOOGLE_VISION: return []
    if google_json_path and os.path.exists(google_json_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_json_path
        
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        context = vision.ImageContext(language_hints=["zh-Hant"])
        response = client.document_text_detection(image=image, image_context=context)
        
        if not response.full_text_annotation: return []
        
        symbols = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for s in word.symbols:
                            if not is_hanja(s.text): continue
                            v = s.bounding_box.vertices
                            x, y = [p.x for p in v], [p.y for p in v]
                            symbols.append({
                                'text': s.text,
                                'center_x': (min(x)+max(x))/2, 'center_y': (min(y)+max(y))/2,
                                'min_x': min(x), 'max_x': max(x), 'min_y': min(y), 'max_y': max(y),
                                'width': max(x)-min(x), 'height': max(y)-min(y),
                                'confidence': s.confidence, 'source': 'Google'
                            })
        
        original_count = len(symbols)
        symbols = merge_google_symbols(symbols, config)
        if len(symbols) < original_count:
            logger.info(f"[OCR] Google 병합: {original_count} -> {len(symbols)}개")
        return symbols
    except Exception as e:
        logger.error(f"[OCR] Google Vision Error: {e}")
        return []

def get_custom_model_ocr(image_path, binary_img, detector, recognizer, config):
    try:
        pil_img = Image.open(image_path).convert("RGB")
        boxes, scores = detector.detect(pil_img)
        if not boxes: return []
        
        # Merge
        original_count = len(boxes)
        boxes, scores = merge_vertical_fragments(boxes, scores, config)
        if len(boxes) < original_count:
            logger.info(f"[OCR] Custom 병합: {original_count} -> {len(boxes)}개")
            
        # Stats
        all_heights = [b[3] for b in boxes]
        all_widths = [b[2] for b in boxes]
        median_h = np.median(all_heights) if all_heights else 0
        median_w = np.median(all_widths) if all_widths else 0
        
        # Recognize
        crops = [pil_img.crop((int(b[0]), int(b[1]), int(b[0]+b[2]), int(b[1]+b[3]))) for b in boxes]
        chars = recognizer(crops) if crops else []
        
        # Filter & Mask (Config values)
        symbols = []
        img_h, _ = binary_img.shape
        ft = config['filtering_thresholds']
        it = config['ink_detection_thresholds']
        
        for char, (x, y, w, h), score in zip(chars, boxes, scores):
            if not char or char == "■": continue
            
            box_dict = {'min_x': x, 'min_y': y, 'max_x': x+w, 'max_y': y+h}
            density = calculate_pixel_density(binary_img, box_dict)
            
            # Hard Filters
            if score < ft['min_score_hard'] or density < ft['density_min_hard']: continue
            # Smart Filters
            if score < ft['smart_score_threshold'] and density < ft['smart_density_threshold']: continue
            
            # Title Removal
            is_huge = (h > median_h * 3.5) if median_h > 0 else False
            is_top = (y < img_h * 0.15) and (h > median_h * 2.5 or w > median_w * 2.5) if median_h > 0 else False
            if median_h > 0 and (is_huge or is_top): continue
            
            # Masking
            final_text, final_type = char, 'TEXT'
            if density >= it['density_ink_heavy']:
                final_text, final_type = '[MASK1]', 'MASK1'
            elif density >= it['density_ink_partial']:
                final_text, final_type = '[MASK2]', 'MASK2'
            else:
                if not is_hanja(char): continue
                
            symbols.append({
                'text': final_text, 'type': final_type,
                'center_x': x+w/2, 'center_y': y+h/2,
                'min_x': x, 'max_x': x+w, 'min_y': y, 'max_y': y+h,
                'width': w, 'height': h,
                'confidence': float(score), 'source': 'Custom', 'density': density
            })
        
        logger.info(f"[OCR] Custom Model 완료: {len(symbols)}개")
        return symbols
    except Exception as e:
        logger.error(f"[OCR] Custom Model Error: {e}")
        return []

# ================================================================================
# Ensemble Reconstruction (Full Logic from Script)
# ================================================================================
def ensemble_reconstruction(google_syms, custom_syms, binary_img, config):
    logger.info("[ENSEMBLE] 앙상블 재구성 시작...")
    img_h, img_w = binary_img.shape
    ec = config['ensemble_config']
    ft = config['filtering_thresholds']
    it = config['ink_detection_thresholds']
    
    # --- Helper Functions ---
    def filter_excessive_masks(nodes):
        filtered, buffer = [], []
        threshold = ec['excessive_mask_threshold']
        for node in nodes:
            if 'MASK' in node.get('type', 'TEXT'): buffer.append(node)
            else:
                if buffer:
                    if len(buffer) < threshold: filtered.extend(buffer)
                    buffer = []
                filtered.append(node)
        if buffer and len(buffer) < threshold: filtered.extend(buffer)
        return filtered

    def merge_split_masks(nodes, avg_h):
        if not nodes: return []
        merged, skip = [], False
        for i in range(len(nodes)):
            if skip: skip = False; continue
            curr = nodes[i]
            if i == len(nodes)-1: merged.append(curr); break
            
            next_node = nodes[i+1]
            if 'MASK' in curr.get('type','TEXT') and 'MASK' in next_node.get('type','TEXT'):
                combined_h = next_node['max_y'] - curr['min_y']
                if combined_h < avg_h * 1.8:
                    new_node = curr.copy()
                    new_node.update({'max_y': next_node['max_y'], 'height': next_node['max_y'] - curr['min_y']})
                    density = calculate_pixel_density(binary_img, new_node)
                    new_node['density'] = density
                    
                    if density < ft['density_min_hard']:
                        skip = True; continue
                    
                    m_type = 'MASK1' if density >= it['density_ink_heavy'] else 'MASK2'
                    new_node.update({'type': m_type, 'text': f'[{m_type}]'})
                    merged.append(new_node)
                    skip = True
                    continue
            merged.append(curr)
        return merged

    def resolve_overlaps(boxes):
        if not boxes: return []
        boxes.sort(key=lambda x: x['min_y'])
        for i in range(len(boxes)-1):
            curr, next_box = boxes[i], boxes[i+1]
            if min(curr['max_x'], next_box['max_x']) - max(curr['min_x'], next_box['min_x']) <= 0: continue
            
            if curr['max_y'] > next_box['min_y']:
                mid_y = (curr['max_y'] + next_box['min_y']) / 2
                curr['max_y'], curr['height'] = mid_y, mid_y - curr['min_y']
                next_box['min_y'], next_box['height'] = mid_y, next_box['max_y'] - mid_y
        return boxes

    def filter_google_overlaps(g_boxes, c_boxes):
        if not g_boxes: return c_boxes
        filtered = []
        for c in c_boxes:
            is_dup = False
            for g in g_boxes:
                dx = abs(c['center_x'] - g['center_x'])
                dy = abs(c['center_y'] - g['center_y'])
                # MASK is preserved even if overlapping
                if 'MASK' in c.get('type', 'TEXT'): pass
                elif (min(c['max_x'], g['max_x']) > max(c['min_x'], g['min_x']) and 
                      min(c['max_y'], g['max_y']) > max(c['min_y'], g['min_y'])) or \
                     (dx < g['width']*0.4 and dy < g['height']*0.4):
                    is_dup = True; break
            if not is_dup: filtered.append(c)
        return filtered

    def infer_gaps(col, step_y, avg_w):
        if not col: return []
        col.sort(key=lambda s: s['center_y'])
        filled = []
        for i, curr in enumerate(col):
            if i > 0:
                prev = col[i-1]
                gap = curr['center_y'] - prev['center_y']
                if gap > step_y * ec['gap_inference_ratio']:
                    missing = int(round(gap/step_y)) - 1
                    if missing > 0:
                        step = gap / (missing + 1)
                        for k in range(1, missing + 1):
                            ny = prev['center_y'] + k*step
                            nb = {'min_x': curr['center_x'] - avg_w/2, 'max_x': curr['center_x'] + avg_w/2,
                                  'min_y': max(0, ny - step_y*0.4), 'max_y': min(img_h, ny + step_y*0.4)}
                            nb.update({'height': nb['max_y']-nb['min_y'], 'width': nb['max_x']-nb['min_x'],
                                       'center_x': (nb['min_x']+nb['max_x'])/2, 'center_y': (nb['min_y']+nb['max_y'])/2})
                            
                            d = calculate_pixel_density(binary_img, nb)
                            if d < ft['density_min_hard']: continue
                            
                            mt = 'MASK1' if d >= it['density_ink_heavy'] else 'MASK2'
                            nb.update({'text': f'[{mt}]', 'type': mt, 'density': d, 'confidence': 0.0, 'source': 'Inferred'})
                            filled.append(nb)
            filled.append(curr)
        return filled

    def check_ink_on_google(g_syms):
        filtered = []
        for s in g_syms:
            d = calculate_pixel_density(binary_img, s)
            s['density'] = d
            if d >= it['density_ink_heavy']: s.update({'type': 'MASK1', 'text': '[MASK1]'})
            elif d >= it['density_ink_partial']: s.update({'type': 'MASK2', 'text': '[MASK2]'})
            elif d < ft['density_min_hard']: continue # Hallucination check
            else: s['type'] = 'TEXT'
            filtered.append(s)
        return filtered

    # --- Preprocessing ---
    all_h = ([s['height'] for s in google_syms] + [s['height'] for s in custom_syms])
    median_h = np.median(all_h) if all_h else 30.0
    
    # Filter Height & Check Ink
    if google_syms:
        google_syms = [s for s in google_syms if not (s['height'] > median_h*2.0)]
        google_syms = check_ink_on_google(google_syms)
    if custom_syms:
        custom_syms = [s for s in custom_syms if not (s['height'] > median_h*3.5)]

    # Resize & Filter Custom
    avg_w = np.mean([s['width'] for s in google_syms]) if google_syms else 0
    median_w = np.median([s['width'] for s in google_syms]) if google_syms else 0
    
    processed_custom = []
    for s in custom_syms:
        if 'MASK' in s.get('type', 'TEXT'):
            processed_custom.append(s); continue
        
        if (s['width']*s['height'] > (median_w*median_h)*0.2 and 
            s['width'] > median_w*0.3 and s['height'] > median_h*0.3):
            
            # Resize logic
            if s['width'] < median_w*0.8 or s['height'] < median_h*0.8:
                tw = max(s['width'], median_w*0.9)
                th = max(s['height'], median_h*0.9)
                cx, cy = s['center_x'], s['center_y']
                s.update({'min_x': max(0, cx-tw/2), 'max_x': min(img_w, cx+tw/2),
                          'min_y': max(0, cy-th/2), 'max_y': min(img_h, cy+th/2)})
                s.update({'width': s['max_x']-s['min_x'], 'height': s['max_y']-s['min_y']})
            processed_custom.append(s)
            
    custom_syms = filter_google_overlaps(google_syms, processed_custom)
    
    if not google_syms and not custom_syms: return [], []

    # --- Column Grouping ---
    all_syms = google_syms + custom_syms
    columns = []
    if all_syms:
        for s in sorted(all_syms, key=lambda x: -x['center_x']):
            found = False
            for col in columns:
                cx = sum(c['center_x'] for c in col) / len(col)
                if abs(s['center_x'] - cx) < (avg_w if avg_w else s['width']) * ec['column_grouping_ratio']:
                    col.append(s); found = True; break
            if not found: columns.append([s])

    # Vertical Step Calculation
    global_steps = []
    for col in columns:
        col.sort(key=lambda s: s['center_y'])
        for k in range(len(col)-1):
            step = col[k+1]['center_y'] - col[k]['center_y']
            if median_h * 0.8 < step < median_h * 1.5: global_steps.append(step)
    global_step = np.median(global_steps) if global_steps else median_h * 1.1

    # --- Reconstruction ---
    final_boxes, lines = [], []
    for col in columns:
        col.sort(key=lambda s: s['center_y'])
        local_steps = [col[k+1]['center_y'] - col[k]['center_y'] for k in range(len(col)-1) 
                       if median_h*0.8 < (col[k+1]['center_y'] - col[k]['center_y']) < median_h*1.5]
        step_y = np.median(local_steps) if local_steps else global_step
        
        # Deduplication in column
        unique_col = [col[0]] if col else []
        for k in range(1, len(col)):
            curr, prev = col[k], unique_col[-1]
            dist = abs(curr['center_y'] - prev['center_y'])
            if dist < median_h * 0.6:
                # Mask priority logic
                pm = 'MASK' in prev.get('type', 'TEXT')
                cm = 'MASK' in curr.get('type', 'TEXT')
                if pm and cm: 
                    if prev['density'] < curr['density']: unique_col[-1] = curr
                elif pm: pass
                elif cm: unique_col[-1] = curr
                elif curr.get('source') == 'Google': unique_col[-1] = curr
                elif prev.get('source') == 'Google': pass
                else: unique_col.append(curr)
            else: unique_col.append(curr)
        
        col = infer_gaps(unique_col, step_y, avg_w if avg_w else median_h)
        
        # Gap Filling with Masks
        filled_col, cy = [], col[0]['min_y'] if col else 0
        for item in col:
            gap = item['min_y'] - cy
            if gap > step_y * 1.2:
                mb = {'min_x': item['center_x'] - (avg_w if avg_w else median_h)/2,
                      'max_x': item['center_x'] + (avg_w if avg_w else median_h)/2,
                      'min_y': max(0, cy + gap*0.1), 'max_y': min(img_h, item['min_y'] - gap*0.1)}
                d = calculate_pixel_density(binary_img, mb)
                if d >= ft['density_min_hard']:
                    mt = 'MASK1' if d >= it['density_ink_heavy'] else 'MASK2'
                    if d >= it['density_ink_partial']:
                        filled_col.append({'text': f'[{mt}]', 'type': mt, 'density': d, 
                                           'min_x': mb['min_x'], 'max_x': mb['max_x'], 
                                           'min_y': mb['min_y'], 'max_y': mb['max_y'], 
                                           'confidence': 0.0, 'source': 'GapFill'})
            
            if item.get('density', 0) < ft['density_min_hard'] and 'MASK' not in item.get('type','TEXT'):
                cy = item['max_y']; continue
                
            filled_col.append(item)
            cy = item['max_y']
            
        filled_col = merge_split_masks(filled_col, median_h)
        filled_col = filter_excessive_masks(filled_col)
        filled_col = resolve_overlaps(filled_col)
        
        final_boxes.extend(filled_col)
        lines.append("".join([s['text'] for s in filled_col]))
        
    logger.info(f"[ENSEMBLE] 완료: {len(final_boxes)}개 박스, {len(lines)}개 열")
    return final_boxes, lines

# ================================================================================
# OCREngine Class
# ================================================================================
class OCREngine:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_ocr_config(config_path)
        
        # Load paths from env
        base_path = os.getenv('OCR_WEIGHTS_BASE_PATH', '/Users/jincerity/Desktop/ocr_weight')
        self.det_ckpt = os.path.join(base_path, os.getenv('OCR_DETECTION_MODEL', 'best.pth'))
        self.rec_ckpt = os.path.join(base_path, os.getenv('OCR_RECOGNITION_MODEL', 'best_5000.pt'))
        self.google_json = os.path.join(base_path, os.getenv('GOOGLE_CREDENTIALS_JSON', 'tidy-node-479900-m7-a4e08301ce8e.json'))
        
        if os.path.exists(self.google_json):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_json
        
        # Device
        dev_cfg = self.config['model_config']['device']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if dev_cfg == 'auto' else torch.device(dev_cfg)
        self.detector = None
        self.recognizer = None

    def _load_models(self):
        if not self.detector:
            self.detector = TextDetector(self.device, self.det_ckpt, self.config)
        if not self.recognizer:
            self.recognizer = ResnetCustom(weight_fn=self.rec_ckpt)
            self.recognizer.to(self.device)

    def run_ocr(self, image_path: str) -> Dict:
        try:
            self._load_models()
            
            # 1. Preprocessing (Exact Match to v12 Script)
            img_bgr = cv2.imread(image_path)
            if img_bgr is None: raise ValueError(f"Image not found: {image_path}")
            
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.medianBlur(img_gray, 3)
            _, img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
            
            # 2. Google Vision
            with io.open(image_path, 'rb') as f: content = f.read()
            google_syms = get_google_ocr(content, self.config, self.google_json)
            
            # 3. Custom Model
            custom_syms = get_custom_model_ocr(image_path, img_binary, self.detector, self.recognizer, self.config)
            
            # 4. Ensemble
            final_boxes, result_lines = ensemble_reconstruction(google_syms, custom_syms, img_binary, self.config)
            
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
            logger.error(f"[OCR] Execution Failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

# ================================================================================
# Global Accessor
# ================================================================================
_engine = None

def get_ocr_engine(config_path: Optional[str] = None) -> OCREngine:
    global _engine
    if _engine is None: _engine = OCREngine(config_path)
    return _engine

def ocr_and_detect(image_path: str, config_path: Optional[str] = None, bbox: Optional[Tuple[int, int, int, int]] = None, device: str = "cuda") -> Dict:
    return get_ocr_engine(config_path).run_ocr(image_path)
