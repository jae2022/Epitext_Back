"""
탁본 관련 API 라우트
"""
from flask import Blueprint, request, jsonify, send_file, current_app
from models import db, Rubbing, RubbingDetail, RubbingStatistics, RestorationTarget, Candidate
from utils.status_calculator import calculate_status, calculate_damage_level
from utils.image_processor import save_uploaded_image
from werkzeug.utils import secure_filename
from ai_modules.preprocessor_unified import preprocess_image_unified
from ai_modules.ocr_engine import get_ocr_engine
from ai_modules.nlp_engine import get_nlp_engine
from ai_modules.swin_engine import get_swin_engine
from unicodedata import normalize
import os
from datetime import datetime
import json
import logging
import cv2
from decimal import Decimal
import re

logger = logging.getLogger(__name__)

rubbings_bp = Blueprint('rubbings', __name__)


def combine_mlm_and_swin(mlm_results, swin_results):
    """
    MLM과 Swin 결과를 합쳐서 최종 복원 후보를 생성합니다.
    
    Args:
        mlm_results: NLP 엔진의 MLM 예측 결과 (order별 top_20 리스트)
        swin_results: Swin 엔진의 복원 결과 (order별 top_20 리스트)
    
    Returns:
        order별 최종 후보 딕셔너리 {order: [candidate_dict, ...]}
    """
    combined = {}
    
    # Swin 결과를 order별로 인덱싱
    swin_by_order = {}
    for item in swin_results.get('results', []):
        order = item.get('order', -1)
        if order >= 0:
            swin_by_order[order] = item.get('top_20', [])
    
    # MLM 결과를 order별로 처리
    for mlm_item in mlm_results.get('results', []):
        order = mlm_item.get('order', -1)
        if order < 0:
            continue
        
        mlm_top20 = mlm_item.get('top_20', [])
        swin_top20 = swin_by_order.get(order, [])
        
        # MLM top-1과 Swin top-1 추출
        mlm_top1_char = mlm_top20[0].get('token', '') if mlm_top20 else None
        swin_top1_char = swin_top20[0].get('token', '') if swin_top20 else None
        
        # 교집합 계산: MLM과 Swin 둘 다 있는 후보 (각각 20개씩 비교)
        mlm_chars = {pred.get('token', ''): pred for pred in mlm_top20}
        swin_chars = {pred.get('token', ''): pred for pred in swin_top20}
        intersection_chars = set(mlm_chars.keys()) & set(swin_chars.keys())
        
        candidates = []
        
        # 1. 교집합 후보들 (F1 Score 계산)
        for char in intersection_chars:
            mlm_pred = mlm_chars[char]
            swin_pred = swin_chars[char]
            
            context_match = mlm_pred.get('probability', 0) * 100
            stroke_match = swin_pred.get('probability', 0) * 100
            
            # F1 Score = 2 * (precision * recall) / (precision + recall)
            # precision = stroke_match, recall = context_match
            if stroke_match + context_match > 0:
                reliability = 2 * (stroke_match * context_match) / (stroke_match + context_match)
            else:
                reliability = 0
            
            candidates.append({
                'character': char,
                'stroke_match': stroke_match,
                'context_match': context_match,
                'reliability': reliability,
                'model_type': 'both',
                'rank_vision': swin_top20.index(swin_pred) + 1 if swin_pred in swin_top20 else None,
                'rank_nlp': mlm_top20.index(mlm_pred) + 1 if mlm_pred in mlm_top20 else None
            })
        
        # 2. MLM만 있는 후보들 (문맥 일치도 우선)
        for char, mlm_pred in mlm_chars.items():
            if char not in intersection_chars:
                context_match = mlm_pred.get('probability', 0) * 100
                candidates.append({
                    'character': char,
                    'stroke_match': None,
                    'context_match': context_match,
                    'reliability': context_match,  # 문맥 일치도가 전체 신뢰도
                    'model_type': 'nlp',
                    'rank_vision': None,
                    'rank_nlp': mlm_top20.index(mlm_pred) + 1 if mlm_pred in mlm_top20 else None
                })
        
        # 3. Swin만 있는 후보들 제거 (NLP가 20개를 뽑았으므로 Swin Only는 필요 없음)
        # 교집합이 없을 때는 NLP Only만 사용하므로, DB에도 Swin Only를 저장하지 않음
        # for char, swin_pred in swin_chars.items():
        #     if char not in intersection_chars:
        #         stroke_match = swin_pred.get('probability', 0) * 100
        #         candidates.append({
        #             'character': char,
        #             'stroke_match': stroke_match,
        #             'context_match': None,
        #             'reliability': stroke_match,  # 획 일치도가 전체 신뢰도
        #             'model_type': 'vision',
        #             'rank_vision': swin_top20.index(swin_pred) + 1 if swin_pred in swin_top20 else None,
        #             'rank_nlp': None
        #         })
        
        # 신뢰도 기준으로 정렬
        candidates.sort(key=lambda x: x['reliability'], reverse=True)
        
        combined[order] = candidates
    
    return combined


def save_results_to_db(rubbing_id, ocr_result, nlp_result, swin_result, combined_candidates, swin_path, start_time):
    """
    AI 처리 결과를 DB에 저장합니다.
    좌표 기반 추측 대신 텍스트 라인을 기준으로 순서대로 MASK를 찾아 매핑합니다.
    
    Args:
        rubbing_id: Rubbing ID
        ocr_result: OCR 결과
        nlp_result: NLP 결과
        swin_result: Swin 결과
        combined_candidates: 복원 로직으로 합쳐진 후보들
        swin_path: Swin 이미지 경로 (크롭용)
        start_time: 처리 시작 시간 (datetime)
    """
    try:
        from datetime import datetime
        
        ocr_results = ocr_result.get('results', [])
        
        # 1. 텍스트 라인 구성 (좌표 기준으로 정렬하여 줄바꿈)
        # Y좌표로 행을 나누고, 각 행 내부에서 X좌표로 정렬
        lines_map = {}
        row_height = 50  # 행 높이 기준 (픽셀)
        
        for item in ocr_results:
            cy = item.get('center_y', 0)
            row_idx = int(cy // row_height)
            if row_idx not in lines_map:
                lines_map[row_idx] = []
            lines_map[row_idx].append(item)
            
        sorted_row_indices = sorted(lines_map.keys())
        text_lines = []
        
        # MASK 아이템과 그 위치를 추적하기 위한 매핑 리스트
        # 구조: list of (row_index, char_index, item_dict)
        mask_map_list = []
        
        for r_idx, row_key in enumerate(sorted_row_indices):
            # X좌표 순으로 정렬
            row_items = sorted(lines_map[row_key], key=lambda x: x.get('center_x', 0))
            
            line_str = ""
            char_counter = 0
            
            for item in row_items:
                text = item.get('text', '')
                item_type = item.get('type', 'TEXT')
                
                # MASK인 경우 추적 리스트에 추가
                if 'MASK' in item_type or 'MASK' in text:
                    # 현재 줄의 현재 글자 위치 저장
                    mask_map_list.append({
                        'ocr_order': item.get('order'),  # OCR 원본 순서
                        'row_index': r_idx,              # 0부터 시작하는 행 번호
                        'char_index': char_counter,      # 0부터 시작하는 글자 번호
                        'item': item
                    })
                
                line_str += text
                char_counter += 1
            
            text_lines.append(line_str)
        
        # 2. NLP 결과 텍스트 (구두점 포함 및 줄바꿈 처리)
        text_with_punc_lines = []
        punctuated_text = nlp_result.get('punctuated_text_with_masks', '')
        
        if punctuated_text:
            # [핵심] 사용자가 요청한 표점(. ? ! ,) 및 한자 표점(。 ， 、) 뒤에 강제 개행 추가
            # 정규표현식을 사용하여 표점 뒤에 \n을 삽입합니다.
            # 패턴: 마침표, 물음표, 느낌표, 쉼표 등
            # ([.?!,。，、]) -> 해당 기호 뒤에 줄바꿈 문자 추가
            processed_text = re.sub(r'([.?!,。，、])', r'\1\n', punctuated_text)
            
            # 개행 문자로 분리하여 리스트 생성 (빈 줄 제거)
            text_with_punc_lines = [line.strip() for line in processed_text.split('\n') if line.strip()]
            
        if not text_with_punc_lines:
            # NLP 실패 시 원본 줄바꿈 사용
            text_with_punc_lines = text_lines
        
        # 3. DB 저장: RubbingDetail (KST 사용)
        end_time = datetime.now()
        processing_time_seconds = int((end_time - start_time).total_seconds())
        
        detail = RubbingDetail(
            rubbing_id=rubbing_id,
            text_content=json.dumps(text_lines, ensure_ascii=False),
            text_content_with_punctuation=json.dumps(text_with_punc_lines, ensure_ascii=False),
            font_types=json.dumps(["해서체"]),  # 예시
            total_processing_time=processing_time_seconds
        )
        db.session.add(detail)
        
        # 4. DB 저장: RestorationTarget (정확한 위치 매핑)
        restoration_targets = []
        all_candidates = []
        
        for mask_info in mask_map_list:
            item = mask_info['item']
            order = mask_info['ocr_order']
            
            # 정확히 계산된 행/열 인덱스 사용
            row_idx = mask_info['row_index']
            char_idx = mask_info['char_index']
            
            damage_type = '부분_훼손' if item.get('type') == 'MASK2' else '완전_훼손'
            
            # 크롭 이미지 처리
            cropped_image_url = None
            crop_x = int(item.get('min_x', 0))
            crop_y = int(item.get('min_y', 0))
            crop_w = int(item.get('max_x', 0) - crop_x)
            crop_h = int(item.get('max_y', 0) - crop_y)
            
            if os.path.exists(swin_path) and crop_w > 0 and crop_h > 0:
                try:
                    img = cv2.imread(swin_path)
                    if img is not None:
                        # 좌표 유효성 검사 및 클리핑
                        h_img, w_img = img.shape[:2]
                        x1 = max(0, min(crop_x, w_img - 1))
                        y1 = max(0, min(crop_y, h_img - 1))
                        x2 = max(x1 + 1, min(crop_x + crop_w, w_img))
                        y2 = max(y1 + 1, min(crop_y + crop_h, h_img))
                        
                        cropped = img[y1:y2, x1:x2]
                        if cropped.size > 0:
                            cropped_dir = os.path.join(os.path.dirname(swin_path), 'cropped')
                            os.makedirs(cropped_dir, exist_ok=True)
                            cropped_filename = f"rubbing_{rubbing_id}_target_{order}.jpg"
                            cropped_path = os.path.join(cropped_dir, cropped_filename)
                            cv2.imwrite(cropped_path, cropped)
                            # URL 경로 수정
                            cropped_image_url = f"/images/rubbings/processed/cropped/{cropped_filename}"
                except Exception as e:
                    logger.warning(f"크롭 실패: {e}")
            
            target = RestorationTarget(
                rubbing_id=rubbing_id,
                row_index=row_idx,
                char_index=char_idx,
                position=f"{row_idx + 1}행 {char_idx + 1}자",  # UI 표시용 1-based index
                damage_type=damage_type,
                cropped_image_url=cropped_image_url,
                crop_x=crop_x, crop_y=crop_y, crop_width=crop_w, crop_height=crop_h
            )
            db.session.add(target)
            db.session.flush()
            
            # 후보 한자 저장 (20개까지 저장)
            candidates = combined_candidates.get(order, [])
            for candidate in candidates[:20]:
                if not candidate or not candidate.get('character'):
                    continue
                
                # Decimal 변환 헬퍼
                def to_decimal(val):
                    return Decimal(str(val)) if val is not None else None
                
                cand_obj = Candidate(
                    target_id=target.id,
                    character=str(candidate['character']),
                    stroke_match=to_decimal(candidate.get('stroke_match')),
                    context_match=to_decimal(candidate.get('context_match')),
                    rank_vision=candidate.get('rank_vision'),
                    rank_nlp=candidate.get('rank_nlp'),
                    model_type=str(candidate.get('model_type', 'nlp')),
                    reliability=to_decimal(candidate.get('reliability'))
                )
                db.session.add(cand_obj)
                all_candidates.append(cand_obj)
        
        # 5. 통계 저장
        total_chars = sum(len(l) for l in text_lines)
        r_count = len(mask_map_list)
        percent = (r_count / total_chars * 100) if total_chars > 0 else 0
        
        stats = RubbingStatistics(
            rubbing_id=rubbing_id,
            total_characters=total_chars,
            restoration_targets=r_count,
            partial_damage=len([m for m in mask_map_list if 'MASK2' in m['item'].get('type', '')]),
            complete_damage=len([m for m in mask_map_list if 'MASK1' in m['item'].get('type', '')]),
            restoration_percentage=Decimal(str(percent))
        )
        db.session.add(stats)
        
        # 6. Rubbing 상태 업데이트
        rubbing = Rubbing.query.get(rubbing_id)
        rubbing.status = calculate_status(processing_time_seconds, percent)
        rubbing.restoration_status = f"{total_chars}자 / 복원 대상 {r_count}자"
        rubbing.processing_time = processing_time_seconds
        rubbing.damage_level = Decimal(str(percent))
        rubbing.processed_at = end_time
        rubbing.inspection_status = "0자 완료"
        
        db.session.commit()
        logger.info(f"[DB] 저장 완료. ID: {rubbing_id}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"[DB] 저장 실패: {e}", exc_info=True)
        raise


@rubbings_bp.route('/api/rubbings', methods=['GET'])
def get_rubbings():
    """탁본 목록 조회 - 프론트엔드 ListPage용 데이터 가공"""
    status = request.args.get('status')
    
    # DB 레벨에서 필터링 (효율성 향상)
    query = Rubbing.query
    
    if status == "completed":
        # is_completed가 True인 것만
        query = query.filter(Rubbing.is_completed == True)
    elif status == "in_progress":
        # is_completed가 False인 것만
        query = query.filter(Rubbing.is_completed == False)
    # else: 전체 조회 (필터링 없음)
    
    # 최신순 정렬
    rubbings = query.order_by(Rubbing.created_at.desc()).all()
    
    results = []
    for idx, r in enumerate(rubbings, 1):
        try:
            stats = r.statistics
            detail = r.details[0] if r.details else None
            
            # 복원 현황 문자열 포맷팅
            total_chars = stats.total_characters if stats else 0
            targets = stats.restoration_targets if stats else 0
            restoration_str = f"{total_chars}자 / 복원 대상 {targets}자" if total_chars > 0 else (r.restoration_status or "-")
            
            # 검수 현황 (InspectionRecord에서 완료된 개수 계산)
            confirmed_count = 0
            if r.restoration_targets:
                inspected_target_ids = {record.target_id for record in r.inspection_records}
                confirmed_count = len(inspected_target_ids)
            inspection_str = r.inspection_status if r.inspection_status else f"{confirmed_count}자 완료"
            
            # 평균 신뢰도 (검수 완료된 것들의 평균)
            avg_reliability = None
            if r.average_reliability is not None:
                avg_reliability = float(r.average_reliability)
            elif r.inspection_records:
                reliabilities = [float(rec.reliability) for rec in r.inspection_records if rec.reliability is not None]
                if reliabilities:
                    avg_reliability = sum(reliabilities) / len(reliabilities)
            
            # 처리 시간 포맷팅
            processing_time_str = "-"
            if r.processing_time:
                minutes = r.processing_time // 60
                seconds = r.processing_time % 60
                if minutes > 0:
                    processing_time_str = f"{minutes}분 {seconds}초"
                else:
                    processing_time_str = f"{seconds}초"
            elif detail and detail.total_processing_time:
                minutes = detail.total_processing_time // 60
                seconds = detail.total_processing_time % 60
                if minutes > 0:
                    processing_time_str = f"{minutes}분 {seconds}초"
                else:
                    processing_time_str = f"{seconds}초"
            
            # 손상 정도
            damage_level_str = "0%"
            if r.damage_level is not None:
                damage_level_str = f"{float(r.damage_level):.1f}%"
            elif stats and stats.damage_level:
                damage_level_str = f"{float(stats.damage_level):.1f}%"
            
            # 처리 일시 (processed_at 우선, 없으면 created_at 사용)
            processed_at = r.processed_at if r.processed_at else r.created_at
            created_at_str = processed_at.strftime('%Y-%m-%d %H:%M') if processed_at else "-"
            
            results.append({
                "id": r.id,
                "index": idx,
                "created_at": created_at_str,
                "filename": r.filename,
                "status": r.status or "처리중",  # "처리중", "우수", "양호", "미흡"
                "restoration_status": restoration_str,
                "processing_time": processing_time_str,
                "damage_level": damage_level_str,
                "inspection_status": inspection_str,
                "average_reliability": f"{avg_reliability:.1f}%" if avg_reliability is not None else "-",
                "is_completed": r.is_completed or False,
                "image_url": r.image_url
            })
        except Exception as e:
            logger.error(f"[API] Rubbing ID {r.id} 처리 중 오류: {e}", exc_info=True)
            # 오류가 발생해도 기본 정보는 반환
            results.append({
                "id": r.id,
                "index": idx,
                "created_at": r.created_at.strftime('%Y-%m-%d %H:%M') if r.created_at else "-",
                "filename": r.filename,
                "status": r.status or "처리중",
                "restoration_status": r.restoration_status or "-",
                "processing_time": "-",
                "damage_level": "0%",
                "inspection_status": r.inspection_status or "0자 완료",
                "average_reliability": "-",
                "is_completed": r.is_completed or False,
                "image_url": r.image_url
            })
    
    return jsonify(results)


@rubbings_bp.route('/api/rubbings/<int:rubbing_id>', methods=['GET'])
def get_rubbing_detail(rubbing_id):
    """탁본 상세 정보 조회 - 프론트엔드 DetailPage용 데이터 보강"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    detail = rubbing.details[0] if rubbing.details else None
    stats = rubbing.statistics
    
    # 폰트 타입 JSON 파싱 (DB에 문자열로 저장된 경우)
    font_types = []
    if detail and detail.font_types:
        try:
            font_types = json.loads(detail.font_types) if isinstance(detail.font_types, str) else detail.font_types
        except:
            font_types = []
    
    # 텍스트 내용 처리
    text_content = []
    text_content_with_punctuation = []
    
    if detail:
        if detail.text_content:
            try:
                text_content = json.loads(detail.text_content) if isinstance(detail.text_content, str) and detail.text_content.startswith('[') else (detail.text_content.split('\n') if detail.text_content else [])
            except:
                text_content = detail.text_content.split('\n') if detail.text_content else []
        
        if detail.text_content_with_punctuation:
            try:
                text_content_with_punctuation = json.loads(detail.text_content_with_punctuation) if isinstance(detail.text_content_with_punctuation, str) and detail.text_content_with_punctuation.startswith('[') else (detail.text_content_with_punctuation.split('\n') if detail.text_content_with_punctuation else [])
            except:
                text_content_with_punctuation = detail.text_content_with_punctuation.split('\n') if detail.text_content_with_punctuation else []
    
    # 처리 일시 포맷팅 (RubbingDetail에는 processed_at이 없으므로 Rubbing에서 가져옴)
    processed_at_str = None
    if rubbing.processed_at:
        processed_at_str = rubbing.processed_at.strftime('%Y-%m-%d %H:%M')
    
    response = {
        "id": rubbing.id,
        "filename": rubbing.filename,
        "image_url": rubbing.image_url,  # 원본 이미지
        "processed_at": processed_at_str,
        "total_processing_time": detail.total_processing_time if detail else (rubbing.processing_time if rubbing.processing_time else 0),
        "font_types": font_types,
        "damage_percentage": float(detail.damage_percentage) if detail and detail.damage_percentage else (float(rubbing.damage_level) if rubbing.damage_level else 0.0),
        
        # 통계 정보
        "statistics": {
            "total_characters": stats.total_characters if stats else 0,
            "restoration_targets": stats.restoration_targets if stats else 0,
            "partial_damage": stats.partial_damage if stats else 0,
            "complete_damage": stats.complete_damage if stats else 0,
            "restoration_percentage": float(stats.restoration_percentage) if stats and stats.restoration_percentage else 0.0
        },
        
        # 텍스트 정보 (구두점 포함)
        "text_content": text_content,
        "text_content_with_punctuation": text_content_with_punctuation,
        
        # 추가 정보
        "created_at": rubbing.created_at.isoformat() if rubbing.created_at else None,
        "updated_at": rubbing.updated_at.isoformat() if rubbing.updated_at else None
    }
    
    return jsonify(response)


@rubbings_bp.route('/api/rubbings/<int:rubbing_id>/download', methods=['GET'])
def download_rubbing(rubbing_id):
    """탁본 원본 파일 다운로드"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    # 이미지 파일 경로 (상대 경로를 절대 경로로 변환)
    image_path = os.path.join(os.getcwd(), rubbing.image_url.lstrip('/'))
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found'}), 404
    
    return send_file(
        image_path,
        as_attachment=True,
        download_name=rubbing.filename
    )


@rubbings_bp.route('/api/rubbings/<int:rubbing_id>/statistics', methods=['GET'])
def get_rubbing_statistics(rubbing_id):
    """탁본 통계 조회"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    statistics = rubbing.statistics
    if not statistics:
        return jsonify({
            'rubbing_id': rubbing_id,
            'total_characters': 0,
            'restoration_targets': 0,
            'partial_damage': 0,
            'complete_damage': 0,
            'restoration_percentage': 0.0
        })
    
    return jsonify(statistics.to_dict())


@rubbings_bp.route('/api/rubbings/<int:rubbing_id>/inspection-status', methods=['GET'])
def get_inspection_status(rubbing_id):
    """검수 상태 조회"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    inspection_records = rubbing.inspection_records
    total_targets = len(rubbing.restoration_targets)
    inspected_count = len(inspection_records)
    
    return jsonify({
        'rubbing_id': rubbing_id,
        'total_targets': total_targets,
        'inspected_count': inspected_count,
        'inspected_targets': [record.to_dict() for record in inspection_records]
    })


@rubbings_bp.route('/api/rubbings/upload', methods=['POST'])
def upload_rubbing():
    """탁본 이미지 업로드 및 전처리"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # 파일명 처리 (한글 파일명 보존)
    original_filename = normalize('NFC', file.filename)
    ext = os.path.splitext(original_filename)[1]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 서버 저장용 영문 파일명
    safe_filename = f"{timestamp}{ext}"
    
    # 경로 설정
    base_folder = current_app.config.get('IMAGES_FOLDER', './images/rubbings')
    original_folder = os.path.join(base_folder, 'original')
    processed_folder = os.path.join(base_folder, 'processed')
    
    # 폴더가 없으면 생성
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    
    # 원본 저장
    original_path = save_uploaded_image(
        file,
        original_folder,
        safe_filename
    )
    
    # 처리 시작 시간 기록 (KST 사용)
    processing_start_time = datetime.now()
    
    # ------------------------------------------------------------------
    # [추가] AI 전처리 모듈 실행 (Integration)
    # ------------------------------------------------------------------
    filename_no_ext = os.path.splitext(safe_filename)[0]
    swin_path = os.path.join(processed_folder, f"swin_{filename_no_ext}.jpg")
    ocr_path = os.path.join(processed_folder, f"ocr_{filename_no_ext}.png")
    
    preprocess_success = False
    preprocess_message = None
    ocr_result = None
    nlp_result = None
    swin_result = None
    combined_candidates = None
    
    try:
        # 1. 통합 전처리 실행
        preprocess_result = preprocess_image_unified(
            input_path=original_path,
            output_swin_path=swin_path,
            output_ocr_path=ocr_path,
            use_rubbing=True  # 탁본 모드 활성화
        )
        
        if preprocess_result.get('success'):
            preprocess_success = True
            logger.info(f"[PREPROCESS] 전처리 성공: {safe_filename}")
            logger.info(f"  - Swin: {swin_path}")
            logger.info(f"  - OCR: {ocr_path}")
            
            # ==================================================================
            # [추가] 2. OCR 엔진 실행 (전처리된 이진 이미지 사용)
            # ==================================================================
            ocr_result = None
            try:
                # 엔진 로드
                ocr_engine = get_ocr_engine()
                
                ocr_result = ocr_engine.run_ocr(ocr_path)
                
                if ocr_result and 'results' in ocr_result:
                    count = len(ocr_result.get('results', []))
                    logger.info(f"[OCR] 분석 완료! 인식된 글자 수: {count}개")
                    
                    # ==================================================================
                    # [추가] 3. NLP 엔진 실행 (구두점 복원 + MLM 예측)
                    # ==================================================================
                    try:
                        nlp_engine = get_nlp_engine()
                        
                        # OCR 결과에서 텍스트 추출
                        ocr_text = ' '.join([item.get('text', '') for item in ocr_result.get('results', [])])
                        
                        # NLP 처리
                        nlp_result = nlp_engine.process_text(
                            raw_text=ocr_text,
                            ocr_results=ocr_result.get('results', []),
                            add_space=True,
                            reduce_punc=True
                        )
                        
                        if nlp_result.get('punctuated_text_with_masks'):
                            logger.info(f"[NLP] 구두점 복원 및 MLM 예측 완료")
                            logger.info(f"  - 마스크 수: {nlp_result.get('statistics', {}).get('total_masks', 0)}개")
                        else:
                            logger.warning("[NLP] NLP 처리 실패")
                            
                    except Exception as nlp_e:
                        logger.error(f"[NLP] 실행 중 예외 발생: {nlp_e}", exc_info=True)
                    # ==================================================================
                    
                    # ==================================================================
                    # [추가] 4. Swin MASK2 복원 실행
                    # ==================================================================
                    try:
                        # Swin 엔진 로드
                        swin_engine = get_swin_engine()
                        
                        # swin_path (전처리된 RGB 이미지)를 사용하여 MASK2 복원
                        swin_result = swin_engine.run_swin_restoration(swin_path, ocr_result)
                        
                        restored_count = len(swin_result.get('results', []))
                        if restored_count > 0:
                            logger.info(f"[SWIN] MASK2 복원 완료: {restored_count}개")
                            
                            stats = swin_result.get('statistics', {})
                            if stats:
                                logger.info(f"  - 평균 신뢰도: {stats.get('top1_probability_avg', 0):.2%}")
                                logger.info(f"  - 최소 신뢰도: {stats.get('top1_probability_min', 0):.2%}")
                                logger.info(f"  - 최대 신뢰도: {stats.get('top1_probability_max', 0):.2%}")
                        else:
                            logger.info("[SWIN] 복원할 MASK2 항목이 없습니다.")
                            
                    except Exception as swin_e:
                        logger.error(f"[SWIN] 실행 중 예외 발생: {swin_e}", exc_info=True)
                    # ==================================================================
                    
                    # ==================================================================
                    # [추가] 5. 복원 로직 실행 (MLM + Swin 합치기)
                    # ==================================================================
                    if nlp_result and swin_result:
                        try:
                            combined_candidates = combine_mlm_and_swin(nlp_result, swin_result)
                            logger.info(f"[RESTORE] 복원 후보 생성 완료: {len(combined_candidates)}개 order")
                        except Exception as restore_e:
                            logger.error(f"[RESTORE] 복원 로직 실행 중 예외 발생: {restore_e}", exc_info=True)
                            combined_candidates = None
                    elif nlp_result:
                        # NLP만 있는 경우: NLP 결과만 사용
                        logger.info("[RESTORE] Swin 결과 없음, NLP 결과만 사용")
                        combined_candidates = {}
                        # NLP 결과에서 후보 추출
                        for item in nlp_result.get('results', []):
                            order = item.get('order', -1)
                            if order >= 0:
                                top20 = item.get('top_20', [])
                                combined_candidates[order] = [
                                    {
                                        'character': pred.get('token', ''),
                                        'stroke_match': None,
                                        'context_match': pred.get('probability', 0) * 100,
                                        'reliability': pred.get('probability', 0) * 100,
                                        'model_type': 'nlp',
                                        'rank_vision': None,
                                        'rank_nlp': idx + 1
                                    }
                                    for idx, pred in enumerate(top20[:20])
                                ]
                    # ==================================================================
                    
                else:
                    error_msg = ocr_result.get('error', 'Unknown Error') if isinstance(ocr_result, dict) else 'OCR 결과 형식 오류'
                    logger.error(f"[OCR] 분석 실패: {error_msg}")
                    
            except Exception as ocr_e:
                logger.error(f"[OCR] 실행 중 예외 발생: {ocr_e}", exc_info=True)
            # ==================================================================

        else:
            preprocess_message = preprocess_result.get('message', 'Unknown error')
            logger.warning(f"[PREPROCESS] 전처리 실패: {preprocess_message}")
            
    except Exception as e:
        preprocess_message = str(e)
        logger.error(f"[PREPROCESS] 전처리 중 치명적 오류: {e}", exc_info=True)
    # ------------------------------------------------------------------
    
    # DB 저장용 URL 수정
    # app.py에서 /images -> ./images/rubbings 로 매핑하므로
    # 실제 파일이 ./images/rubbings/original/safe_filename 에 있다면
    # URL은 /images/original/safe_filename 이어야 함
    image_url = f"/images/original/{safe_filename}"
    
    # DB에 레코드 생성
    rubbing = Rubbing(
        image_url=image_url,
        filename=original_filename,  # 원본 한글 파일명 저장
        status="처리중",
        is_completed=False
    )
    
    db.session.add(rubbing)
    db.session.commit()
    
    # ==================================================================
    # [추가] 6. AI 처리 결과를 DB에 저장
    # ==================================================================
    # OCR 결과만 있어도 최소한의 데이터는 저장
    if ocr_result and 'results' in ocr_result:
        try:
            # combined_candidates가 없으면 빈 딕셔너리로 처리
            if combined_candidates is None:
                combined_candidates = {}
            
            # nlp_result가 없으면 빈 결과로 처리
            if nlp_result is None:
                nlp_result = {
                    'punctuated_text_with_masks': '',
                    'results': [],
                    'statistics': {'total_masks': 0}
                }
            
            # swin_result가 없으면 빈 결과로 처리
            if swin_result is None:
                swin_result = {
                    'results': [],
                    'statistics': {}
                }
            
            save_results_to_db(
                rubbing_id=rubbing.id,
                ocr_result=ocr_result,
                nlp_result=nlp_result,
                swin_result=swin_result,
                combined_candidates=combined_candidates,
                swin_path=swin_path,
                start_time=processing_start_time
            )
            logger.info(f"[DB] AI 처리 결과 저장 완료: Rubbing ID {rubbing.id}")
            
            # 저장 후 Rubbing 레코드 다시 조회하여 최신 상태 반환
            db.session.refresh(rubbing)
        except Exception as db_e:
            logger.error(f"[DB] 저장 중 오류: {db_e}", exc_info=True)
            # DB 저장 실패 시에도 최소한의 상태 업데이트
            try:
                # rubbing 객체를 다시 조회하여 detached 상태 방지
                db.session.rollback()
                rubbing = Rubbing.query.get(rubbing.id)
                if rubbing:
                    rubbing.status = "처리중"
                    rubbing.restoration_status = "처리 실패"
                    db.session.commit()
                    logger.info(f"[DB] 상태 업데이트 완료: 처리 실패로 설정")
            except Exception as status_e:
                logger.error(f"[DB] 상태 업데이트 실패: {status_e}", exc_info=True)
                db.session.rollback()
    else:
        # OCR도 실패한 경우
        logger.error("[DB] OCR 결과가 없어 DB 저장을 건너뜁니다.")
        try:
            rubbing.status = "처리중"
            rubbing.restoration_status = "OCR 실패"
            db.session.commit()
        except:
            pass
    # ==================================================================
    
    # 응답 데이터 준비
    response_data = rubbing.to_dict()
    
    # 전처리 결과 정보 추가 (선택사항)
    if preprocess_success:
        response_data['preprocessing'] = {
            'success': True,
            'swin_path': f"/images/rubbings/processed/swin_{filename_no_ext}.jpg",
            'ocr_path': f"/images/rubbings/processed/ocr_{filename_no_ext}.png"
        }
    elif preprocess_message:
        response_data['preprocessing'] = {
            'success': False,
            'message': preprocess_message
        }
    
    # OCR 결과 추가
    if ocr_result:
        response_data['ocr'] = ocr_result
    
    # NLP 결과 추가
    if nlp_result:
        response_data['nlp'] = nlp_result
    
    # Swin 복원 결과 추가
    if swin_result:
        response_data['swin'] = swin_result
    
    return jsonify(response_data), 201


@rubbings_bp.route('/api/rubbings/complete', methods=['POST'])
def complete_rubbings():
    """복원 완료 처리"""
    data = request.get_json()
    selected_ids = data.get('selected_ids', [])
    
    if not selected_ids:
        return jsonify({'error': 'No IDs provided'}), 400
    
    # 선택된 탁본들의 is_completed를 true로 업데이트
    rubbings = Rubbing.query.filter(Rubbing.id.in_(selected_ids)).all()
    
    for rubbing in rubbings:
        rubbing.is_completed = True
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'completed_count': len(rubbings)
    })

