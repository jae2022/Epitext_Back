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
import os
from datetime import datetime
import json
import logging
import cv2
from decimal import Decimal

logger = logging.getLogger(__name__)

rubbings_bp = Blueprint('rubbings', __name__)


def combine_mlm_and_swin(mlm_results, swin_results):
    """
    MLM과 Swin 결과를 합쳐서 최종 복원 후보를 생성합니다.
    
    Args:
        mlm_results: NLP 엔진의 MLM 예측 결과 (order별 top_10 리스트)
        swin_results: Swin 엔진의 복원 결과 (order별 top_10 리스트)
    
    Returns:
        order별 최종 후보 딕셔너리 {order: [candidate_dict, ...]}
    """
    combined = {}
    
    # Swin 결과를 order별로 인덱싱
    swin_by_order = {}
    for item in swin_results.get('results', []):
        order = item.get('order', -1)
        if order >= 0:
            swin_by_order[order] = item.get('top_10', [])
    
    # MLM 결과를 order별로 처리
    for mlm_item in mlm_results.get('results', []):
        order = mlm_item.get('order', -1)
        if order < 0:
            continue
        
        mlm_top10 = mlm_item.get('top_10', [])
        swin_top10 = swin_by_order.get(order, [])
        
        # MLM top-1과 Swin top-1 추출
        mlm_top1_char = mlm_top10[0].get('token', '') if mlm_top10 else None
        swin_top1_char = swin_top10[0].get('token', '') if swin_top10 else None
        
        # 교집합 계산: MLM과 Swin 둘 다 있는 후보
        mlm_chars = {pred.get('token', ''): pred for pred in mlm_top10}
        swin_chars = {pred.get('token', ''): pred for pred in swin_top10}
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
                'rank_vision': swin_top10.index(swin_pred) + 1 if swin_pred in swin_top10 else None,
                'rank_nlp': mlm_top10.index(mlm_pred) + 1 if mlm_pred in mlm_top10 else None
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
                    'rank_nlp': mlm_top10.index(mlm_pred) + 1 if mlm_pred in mlm_top10 else None
                })
        
        # 3. Swin만 있는 후보들 (획 일치도만)
        for char, swin_pred in swin_chars.items():
            if char not in intersection_chars:
                stroke_match = swin_pred.get('probability', 0) * 100
                candidates.append({
                    'character': char,
                    'stroke_match': stroke_match,
                    'context_match': None,
                    'reliability': stroke_match,  # 획 일치도가 전체 신뢰도
                    'model_type': 'vision',
                    'rank_vision': swin_top10.index(swin_pred) + 1 if swin_pred in swin_top10 else None,
                    'rank_nlp': None
                })
        
        # 신뢰도 기준으로 정렬
        candidates.sort(key=lambda x: x['reliability'], reverse=True)
        
        combined[order] = candidates
    
    return combined


def save_results_to_db(rubbing_id, ocr_result, nlp_result, swin_result, combined_candidates, swin_path, start_time):
    """
    AI 처리 결과를 DB에 저장합니다.
    
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
        # 1. OCR 결과에서 텍스트 추출
        ocr_results = ocr_result.get('results', [])
        text_lines = []
        text_with_punc_lines = []
        
        # OCR 결과를 행별로 그룹화 (항상 rows 변수 정의)
        rows = {}
        if not ocr_results:
            logger.warning("[DB] OCR 결과가 비어있습니다.")
            text_lines = ['']
        else:
            for item in ocr_results:
                # center_y 기준으로 행 분류 (간단한 휴리스틱)
                y = item.get('center_y', 0)
                row_key = int(y // 50)  # 50px 단위로 행 분류
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append(item)
            
            # 행별로 정렬하고 텍스트 추출
            if rows:
                for row_idx, row_key in enumerate(sorted(rows.keys())):
                    row_items = sorted(rows[row_key], key=lambda x: x.get('center_x', 0))
                    text_line = ''.join([item.get('text', '') for item in row_items])
                    if text_line:  # 빈 줄 제외
                        text_lines.append(text_line)
            
            if not text_lines:
                text_lines = ['']
        
        # 2. 구두점 복원 텍스트 추출
        punctuated_text = nlp_result.get('punctuated_text_with_masks', '')
        if punctuated_text:
            text_with_punc_lines = [line for line in punctuated_text.split('\n') if line.strip()]
        
        if not text_with_punc_lines:
            # 구두점 복원 텍스트가 없으면 OCR 텍스트 사용
            text_with_punc_lines = text_lines
        
        # 처리 시간 계산
        end_time = datetime.utcnow()
        processing_time_seconds = int((end_time - start_time).total_seconds())
        
        # 3. RubbingDetail 저장
        detail = RubbingDetail(
            rubbing_id=rubbing_id,
            text_content=json.dumps(text_lines, ensure_ascii=False),
            text_content_with_punctuation=json.dumps(text_with_punc_lines, ensure_ascii=False),
            font_types=json.dumps([]),  # TODO: 폰트 타입 분석 추가
            damage_percentage=None,  # 통계에서 계산
            total_processing_time=processing_time_seconds
        )
        db.session.add(detail)
        
        # 4. 복원 대상 및 후보 추출
        restoration_targets = []
        all_candidates = []
        
        # OCR 결과에서 MASK1, MASK2 찾기
        mask_items = [item for item in ocr_results if 'MASK' in item.get('type', '')]
        
        if not mask_items:
            logger.info("[DB] 복원 대상(MASK)이 없습니다.")
        
        for mask_item in mask_items:
            order = mask_item.get('order', -1)
            if order < 0:
                continue
            
            # 행/열 인덱스 계산 (간단한 휴리스틱)
            y = mask_item.get('center_y', 0)
            x = mask_item.get('center_x', 0)
            row_idx = int(y // 50)
            char_idx = len([item for item in rows.get(int(y // 50), []) if item.get('center_x', 0) < x])
            
            damage_type = '부분_훼손' if mask_item.get('type') == 'MASK2' else '완전_훼손'
            
            # 크롭 이미지 생성 (선택사항)
            cropped_image_url = None
            crop_x = int(mask_item.get('min_x', 0))
            crop_y = int(mask_item.get('min_y', 0))
            crop_w = int(mask_item.get('max_x', 0) - mask_item.get('min_x', 0))
            crop_h = int(mask_item.get('max_y', 0) - mask_item.get('min_y', 0))
            
            # 크롭 이미지 저장 (선택사항)
            if os.path.exists(swin_path) and crop_w > 0 and crop_h > 0:
                try:
                    img = cv2.imread(swin_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        x1 = max(0, min(crop_x, w - 1))
                        y1 = max(0, min(crop_y, h - 1))
                        x2 = max(x1 + 1, min(crop_x + crop_w, w))
                        y2 = max(y1 + 1, min(crop_y + crop_h, h))
                        
                        cropped = img[y1:y2, x1:x2]
                        if cropped.size > 0:
                            cropped_dir = os.path.join(os.path.dirname(swin_path), 'cropped')
                            os.makedirs(cropped_dir, exist_ok=True)
                            cropped_filename = f"rubbing_{rubbing_id}_target_{order}.jpg"
                            cropped_path = os.path.join(cropped_dir, cropped_filename)
                            cv2.imwrite(cropped_path, cropped)
                            cropped_image_url = f"/images/rubbings/processed/cropped/{cropped_filename}"
                except Exception as e:
                    logger.warning(f"[DB] 크롭 이미지 저장 실패: {e}")
            
            target = RestorationTarget(
                rubbing_id=rubbing_id,
                row_index=row_idx,
                char_index=char_idx,
                position=f"{row_idx + 1}행 {char_idx + 1}자",
                damage_type=damage_type,
                cropped_image_url=cropped_image_url,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_width=crop_w,
                crop_height=crop_h
            )
            db.session.add(target)
            db.session.flush()  # ID 생성
            
            restoration_targets.append((target, order))
            
            # 후보 저장
            candidates = combined_candidates.get(order, [])
            for rank, candidate in enumerate(candidates[:10]):  # Top-10만 저장
                # 후보 데이터 검증
                if not candidate or not candidate.get('character'):
                    continue
                
                try:
                    # 안전한 Decimal 변환
                    def safe_decimal(value, default=0):
                        if value is None:
                            return None if default is None else Decimal(str(default))
                        try:
                            return Decimal(str(float(value)))
                        except (ValueError, TypeError):
                            return Decimal(str(default))
                    
                    candidate_obj = Candidate(
                        target_id=target.id,
                        character=str(candidate['character'])[:10],  # 최대 10자
                        stroke_match=safe_decimal(candidate.get('stroke_match'), None),
                        context_match=safe_decimal(candidate.get('context_match'), 0),
                        rank_vision=candidate.get('rank_vision'),
                        rank_nlp=candidate.get('rank_nlp'),
                        model_type=str(candidate.get('model_type', 'nlp'))[:10],
                        reliability=safe_decimal(candidate.get('reliability'), 0)
                    )
                    db.session.add(candidate_obj)
                    all_candidates.append(candidate_obj)
                except Exception as cand_e:
                    logger.warning(f"[DB] 후보 저장 실패 (order={order}, rank={rank}): {cand_e}", exc_info=True)
                    continue
        
        # 5. RubbingStatistics 저장
        total_chars = sum(len(line) for line in text_lines)
        restoration_count = len(restoration_targets)
        partial_count = len([t for t, _ in restoration_targets if t.damage_type == '부분_훼손'])
        complete_count = len([t for t, _ in restoration_targets if t.damage_type == '완전_훼손'])
        restoration_percentage = (restoration_count / total_chars * 100) if total_chars > 0 else 0
        
        statistics = RubbingStatistics(
            rubbing_id=rubbing_id,
            total_characters=total_chars,
            restoration_targets=restoration_count,
            partial_damage=partial_count,
            complete_damage=complete_count,
            restoration_percentage=Decimal(str(restoration_percentage))
        )
        db.session.add(statistics)
        
        # Rubbing 레코드 업데이트 (상태, 처리 시간, 손상 정도 등)
        rubbing = Rubbing.query.get(rubbing_id)
        if not rubbing:
            raise ValueError(f"Rubbing ID {rubbing_id}를 찾을 수 없습니다.")
        
        # 손상 정도 계산 (안전한 변환)
        try:
            damage_level = Decimal(str(float(restoration_percentage)))
        except (ValueError, TypeError):
            damage_level = Decimal('0.0')
            logger.warning(f"[DB] 손상 정도 계산 실패, 0.0으로 설정: {restoration_percentage}")
        
        # 상태 계산
        try:
            status = calculate_status(processing_time_seconds, float(damage_level))
        except Exception as status_e:
            logger.warning(f"[DB] 상태 계산 실패, 기본값 사용: {status_e}")
            status = "처리중"  # 기본값
        
        # 복원 현황 문자열 생성
        restoration_status = f"{total_chars}자 / 복원 대상 {restoration_count}자" if total_chars > 0 else "-"
        
        # 처리 시간 포맷팅
        processing_time_str = f"{processing_time_seconds // 60}분 {processing_time_seconds % 60}초"
        
        # 업데이트
        rubbing.status = status
        rubbing.restoration_status = restoration_status
        rubbing.processing_time = processing_time_seconds
        rubbing.damage_level = damage_level
        rubbing.processed_at = end_time
        rubbing.inspection_status = "0자 완료"  # 초기값
        rubbing.average_reliability = None  # 검수 후 계산
        
        db.session.commit()
        logger.info(f"[DB] 저장 완료: RubbingDetail, {restoration_count}개 RestorationTarget, {len(all_candidates)}개 Candidate")
        logger.info(f"[DB] 상태 업데이트: {status}, 처리 시간: {processing_time_seconds}초")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"[DB] 저장 중 오류: {e}", exc_info=True)
        # 상세한 에러 정보 로깅
        import traceback
        logger.error(f"[DB] 상세 에러:\n{traceback.format_exc()}")
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
            
            # 처리 일시
            created_at_str = r.created_at.strftime('%Y-%m-%d %H:%M') if r.created_at else "-"
            
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
    
    # 처리 일시 포맷팅
    processed_at_str = None
    if detail and detail.processed_at:
        processed_at_str = detail.processed_at.strftime('%Y-%m-%d %H:%M')
    elif rubbing.processed_at:
        processed_at_str = rubbing.processed_at.strftime('%Y-%m-%d %H:%M')
    
    response = {
        "id": rubbing.id,
        "filename": rubbing.filename,
        "image_url": rubbing.image_url,  # 원본 이미지
        "processed_at": processed_at_str,
        "total_processing_time": detail.total_processing_time if detail else (rubbing.processing_time if rubbing.processing_time else 0),
        "font_types": font_types,
        "damage_percentage": float(stats.damage_level) if stats and stats.damage_level else 0.0,
        
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
    
    # 파일명 보안 처리
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    
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
        unique_filename
    )
    
    # 처리 시작 시간 기록
    processing_start_time = datetime.utcnow()
    
    # ------------------------------------------------------------------
    # [추가] AI 전처리 모듈 실행 (Integration)
    # ------------------------------------------------------------------
    filename_no_ext = os.path.splitext(unique_filename)[0]
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
            logger.info(f"[PREPROCESS] 전처리 성공: {unique_filename}")
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
                                top10 = item.get('top_10', [])
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
                                    for idx, pred in enumerate(top10[:10])
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
    
    # DB 저장 경로 (원본 이미지 URL)
    image_url = f"/images/rubbings/original/{unique_filename}"
    
    # DB에 레코드 생성
    rubbing = Rubbing(
        image_url=image_url,
        filename=filename,
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

