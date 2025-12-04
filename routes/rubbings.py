"""
탁본 관련 API 라우트
"""
from flask import Blueprint, request, jsonify, send_file, current_app
from models import db, Rubbing, RubbingDetail, RubbingStatistics
from utils.status_calculator import calculate_status, calculate_damage_level
from utils.image_processor import save_uploaded_image
from werkzeug.utils import secure_filename
from ai_modules.preprocessor_unified import preprocess_image_unified
from ai_modules.ocr_engine import get_ocr_engine
from ai_modules.nlp_engine import get_nlp_engine
import os
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

rubbings_bp = Blueprint('rubbings', __name__)


@rubbings_bp.route('/api/rubbings', methods=['GET'])
def get_rubbings():
    """탁본 목록 조회"""
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
    
    return jsonify([rubbing.to_dict() for rubbing in rubbings])


@rubbings_bp.route('/api/rubbings/<int:rubbing_id>', methods=['GET'])
def get_rubbing_detail(rubbing_id):
    """탁본 상세 정보 조회"""
    rubbing = Rubbing.query.get(rubbing_id)
    if not rubbing:
        return jsonify({'error': 'Rubbing not found'}), 404
    
    detail = rubbing.details[0] if rubbing.details else None
    
    if not detail:
        return jsonify({
            'id': rubbing.id,
            'image_url': rubbing.image_url,
            'filename': rubbing.filename,
            'text_content': [],
            'text_content_with_punctuation': [],
            'font_types': [],
            'damage_percentage': None,
            'processed_at': None,
            'total_processing_time': None,
            'created_at': rubbing.created_at.isoformat() if rubbing.created_at else None,
            'updated_at': rubbing.updated_at.isoformat() if rubbing.updated_at else None
        })
    
    detail_dict = detail.to_dict()
    detail_dict['id'] = rubbing.id
    detail_dict['image_url'] = rubbing.image_url
    detail_dict['filename'] = rubbing.filename
    
    return jsonify(detail_dict)


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
    
    # ------------------------------------------------------------------
    # [추가] AI 전처리 모듈 실행 (Integration)
    # ------------------------------------------------------------------
    filename_no_ext = os.path.splitext(unique_filename)[0]
    swin_path = os.path.join(processed_folder, f"swin_{filename_no_ext}.jpg")
    ocr_path = os.path.join(processed_folder, f"ocr_{filename_no_ext}.png")
    
    preprocess_success = False
    preprocess_message = None
    ocr_result = None
    swin_result = None
    
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
<<<<<<< HEAD
                    # [추가] 3. Swin MASK2 복원 실행
                    # ==================================================================
                    try:
                        from ai_modules.swin_engine import get_swin_engine
                        
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
=======
                    # [추가] 3. NLP 처리 (구두점 복원 + MLM 예측)
                    # ==================================================================
                    try:
                        # OCR 결과에서 텍스트 추출 (results 배열에서 text 추출)
                        ocr_results_list = ocr_result.get('results', [])
                        if ocr_results_list:
                            # 텍스트를 순서대로 결합
                            raw_text = "".join([item.get('text', '') for item in ocr_results_list])
                            logger.info(f"[NLP] OCR 텍스트 추출 완료: {len(raw_text)} 글자")
                            
                            # NLP 엔진 로드 및 처리
                            nlp_engine = get_nlp_engine()
                            # OCR 결과를 NLP에 전달하여 order 정보 유지
                            nlp_result = nlp_engine.process_text(raw_text, ocr_results=ocr_results_list)
                            
                            if nlp_result and 'punctuated_text_with_masks' in nlp_result:
                                logger.info(f"[NLP] 처리 완료!")
                                logger.info(f"  - 구두점 복원 텍스트: {len(nlp_result.get('punctuated_text_with_masks', ''))} 글자")
                                logger.info(f"  - MLM 예측 마스크 수: {nlp_result.get('statistics', {}).get('total_masks', 0)}개")
                                
                                # (선택사항) 여기서 nlp_result 데이터를 DB에 저장하는 로직을 추가할 수 있습니다.
                                # 예: save_nlp_results_to_db(rubbing.id, nlp_result)
                            else:
                                error_msg = nlp_result.get('error', 'Unknown Error') if isinstance(nlp_result, dict) else 'Unknown Error'
                                logger.error(f"[NLP] 처리 실패: {error_msg}")
                        else:
                            logger.warning("[NLP] OCR 결과에 텍스트가 없어 NLP 처리를 건너뜁니다.")
                            
                    except Exception as nlp_e:
                        logger.error(f"[NLP] 실행 중 예외 발생: {nlp_e}", exc_info=True)
>>>>>>> main
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

