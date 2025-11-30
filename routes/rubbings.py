"""
탁본 관련 API 라우트
"""
from flask import Blueprint, request, jsonify, send_file
from models import db, Rubbing, RubbingDetail, RubbingStatistics
from utils.status_calculator import calculate_status, calculate_damage_level
from utils.image_processor import save_uploaded_image
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json

rubbings_bp = Blueprint('rubbings', __name__)


@rubbings_bp.route('/api/rubbings', methods=['GET'])
def get_rubbings():
    """탁본 목록 조회"""
    status = request.args.get('status')
    
    query = Rubbing.query
    
    # 필터링
    if status == "복원 완료":
        query = query.filter(Rubbing.is_completed == True)
    elif status == "복원 진행중":
        query = query.filter(Rubbing.is_completed == False)
    
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
    """탁본 이미지 업로드"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # 파일명 보안 처리
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    
    # 이미지 저장
    from flask import current_app
    images_folder = current_app.config.get('IMAGES_FOLDER', './images/rubbings')
    image_path = save_uploaded_image(
        file,
        images_folder,
        unique_filename
    )
    
    # 상대 경로 저장 (API 응답용)
    image_url = f"/images/rubbings/{unique_filename}"
    
    # DB에 레코드 생성
    rubbing = Rubbing(
        image_url=image_url,
        filename=filename,
        status="처리중",
        is_completed=False
    )
    
    db.session.add(rubbing)
    db.session.commit()
    
    return jsonify(rubbing.to_dict()), 201


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

