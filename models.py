from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Rubbing(db.Model):
    """탁본 목록 테이블"""
    __tablename__ = 'rubbings'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    image_url = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(db.String(20))  # "처리중", "우수", "양호", "미흡"
    restoration_status = db.Column(db.String(100))  # "356자 / 복원 대상 23자"
    processing_time = db.Column(db.Integer)  # 초 단위
    damage_level = db.Column(db.Numeric(5, 2))  # 복원 대상 비율 (%)
    inspection_status = db.Column(db.String(100))  # "12자 완료"
    average_reliability = db.Column(db.Numeric(5, 2))  # 평균 신뢰도 (%)
    is_completed = db.Column(db.Boolean, default=False, nullable=False)
    processed_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 관계
    details = db.relationship('RubbingDetail', backref='rubbing', lazy=True, cascade='all, delete-orphan')
    statistics = db.relationship('RubbingStatistics', backref='rubbing', lazy=True, uselist=False, cascade='all, delete-orphan')
    restoration_targets = db.relationship('RestorationTarget', backref='rubbing', lazy=True, cascade='all, delete-orphan')
    inspection_records = db.relationship('InspectionRecord', backref='rubbing', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """API 응답용 딕셔너리 변환"""
        return {
            'id': self.id,
            'image_url': self.image_url,
            'filename': self.filename,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'status': self.status,
            'restoration_status': self.restoration_status,
            'processing_time': self.processing_time,
            'damage_level': float(self.damage_level) if self.damage_level else None,
            'inspection_status': self.inspection_status,
            'average_reliability': float(self.average_reliability) if self.average_reliability else None,
            'is_completed': self.is_completed
        }


class RubbingDetail(db.Model):
    """탁본 상세 정보 테이블"""
    __tablename__ = 'rubbing_details'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rubbing_id = db.Column(db.Integer, db.ForeignKey('rubbings.id', ondelete='CASCADE'), nullable=False)
    text_content = db.Column(db.Text)  # OCR 결과 (JSON 배열 또는 TEXT, 구두점 복원 전)
    text_content_with_punctuation = db.Column(db.Text)  # 구두점 복원 모델 적용 후
    font_types = db.Column(db.String(255))  # JSON 배열: ["행서체", "전서체"]
    damage_percentage = db.Column(db.Numeric(5, 2))
    total_processing_time = db.Column(db.Integer)  # 초 단위
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """API 응답용 딕셔너리 변환"""
        import json
        return {
            'id': self.id,
            'rubbing_id': self.rubbing_id,
            'text_content': json.loads(self.text_content) if self.text_content and self.text_content.startswith('[') else (self.text_content.split('\n') if self.text_content else []),
            'text_content_with_punctuation': json.loads(self.text_content_with_punctuation) if self.text_content_with_punctuation and self.text_content_with_punctuation.startswith('[') else (self.text_content_with_punctuation.split('\n') if self.text_content_with_punctuation else []),
            'font_types': json.loads(self.font_types) if self.font_types else [],
            'damage_percentage': float(self.damage_percentage) if self.damage_percentage else None,
            'total_processing_time': self.total_processing_time,
            'processed_at': self.rubbing.processed_at.isoformat() if self.rubbing.processed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class RubbingStatistics(db.Model):
    """탁본 통계 테이블"""
    __tablename__ = 'rubbing_statistics'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rubbing_id = db.Column(db.Integer, db.ForeignKey('rubbings.id', ondelete='CASCADE'), nullable=False, unique=True)
    total_characters = db.Column(db.Integer, nullable=False)
    restoration_targets = db.Column(db.Integer, nullable=False)
    partial_damage = db.Column(db.Integer)  # 부분 훼손 글자 수
    complete_damage = db.Column(db.Integer)  # 완전 훼손 글자 수
    restoration_percentage = db.Column(db.Numeric(5, 2))  # 복원 비율 (%)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """API 응답용 딕셔너리 변환"""
        return {
            'rubbing_id': self.rubbing_id,
            'total_characters': self.total_characters,
            'restoration_targets': self.restoration_targets,
            'partial_damage': self.partial_damage,
            'complete_damage': self.complete_damage,
            'restoration_percentage': float(self.restoration_percentage) if self.restoration_percentage else None
        }


class RestorationTarget(db.Model):
    """복원 대상 글자 테이블"""
    __tablename__ = 'restoration_targets'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rubbing_id = db.Column(db.Integer, db.ForeignKey('rubbings.id', ondelete='CASCADE'), nullable=False)
    row_index = db.Column(db.Integer, nullable=False)  # 행 인덱스 (0부터 시작)
    char_index = db.Column(db.Integer, nullable=False)  # 글자 인덱스 (0부터 시작)
    position = db.Column(db.String(50))  # "1행 1자" 형식
    damage_type = db.Column(db.String(20))  # "부분_훼손" 또는 "완전_훼손"
    cropped_image_url = db.Column(db.String(255))  # 크롭된 이미지 URL
    crop_x = db.Column(db.Integer)  # 크롭 영역 X 좌표
    crop_y = db.Column(db.Integer)  # 크롭 영역 Y 좌표
    crop_width = db.Column(db.Integer)  # 크롭 영역 너비
    crop_height = db.Column(db.Integer)  # 크롭 영역 높이
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 관계
    candidates = db.relationship('Candidate', backref='target', lazy=True, cascade='all, delete-orphan')
    inspection_records = db.relationship('InspectionRecord', backref='target', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """API 응답용 딕셔너리 변환"""
        return {
            'id': self.id,
            'row_index': self.row_index,
            'char_index': self.char_index,
            'position': self.position,
            'damage_type': self.damage_type
        }


class Candidate(db.Model):
    """후보 한자 테이블"""
    __tablename__ = 'candidates'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    target_id = db.Column(db.Integer, db.ForeignKey('restoration_targets.id', ondelete='CASCADE'), nullable=False)
    character = db.Column(db.String(10), nullable=False)  # 후보 한자
    stroke_match = db.Column(db.Numeric(5, 2))  # 획 일치도 (null 가능)
    context_match = db.Column(db.Numeric(5, 2), nullable=False)  # 문맥 일치도
    rank_vision = db.Column(db.Integer)  # Vision 모델 순위 (null 가능)
    rank_nlp = db.Column(db.Integer, nullable=False)  # NLP 모델 순위
    model_type = db.Column(db.String(10))  # "nlp", "both", "vision"
    reliability = db.Column(db.Numeric(5, 2))  # 최종 신뢰도 (F1 Score)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """API 응답용 딕셔너리 변환"""
        return {
            'id': self.id,
            'character': self.character,
            'stroke_match': float(self.stroke_match) if self.stroke_match is not None else None,
            'context_match': float(self.context_match) if self.context_match is not None else None,
            'rank_vision': self.rank_vision,
            'rank_nlp': self.rank_nlp,
            'model_type': self.model_type,
            'reliability': float(self.reliability) if self.reliability is not None else None
        }


class InspectionRecord(db.Model):
    """검수 기록 테이블"""
    __tablename__ = 'inspection_records'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rubbing_id = db.Column(db.Integer, db.ForeignKey('rubbings.id', ondelete='CASCADE'), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey('restoration_targets.id', ondelete='CASCADE'), nullable=False)
    selected_character = db.Column(db.String(10), nullable=False)  # 선택된 한자
    selected_candidate_id = db.Column(db.Integer, db.ForeignKey('candidates.id', ondelete='SET NULL'))
    reliability = db.Column(db.Numeric(5, 2))  # 선택된 후보의 신뢰도
    inspected_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """API 응답용 딕셔너리 변환"""
        return {
            'target_id': self.target_id,
            'selected_character': self.selected_character,
            'selected_candidate_id': self.selected_candidate_id,
            'inspected_at': self.inspected_at.isoformat() if self.inspected_at else None
        }

