"""
Flask 애플리케이션 진입점
"""
from flask import Flask, send_from_directory
from flask_cors import CORS
from config import Config
from models import db
from routes.rubbings import rubbings_bp
from routes.targets import targets_bp
from routes.inspection import inspection_bp
from routes.translation import translation_bp
import os

def create_app():
    """Flask 앱 생성 및 설정"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # CORS 설정 (프론트엔드와 통신을 위해)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # 데이터베이스 초기화
    db.init_app(app)
    
    # 블루프린트 등록
    app.register_blueprint(rubbings_bp)
    app.register_blueprint(targets_bp)
    app.register_blueprint(inspection_bp)
    app.register_blueprint(translation_bp)
    
    # 필요한 디렉토리 생성
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.IMAGES_FOLDER, exist_ok=True)
    os.makedirs(Config.CROPPED_IMAGES_FOLDER, exist_ok=True)
    
    # 전처리 결과 저장 폴더 생성
    original_folder = os.path.join(Config.IMAGES_FOLDER, 'original')
    processed_folder = os.path.join(Config.IMAGES_FOLDER, 'processed')
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    
    @app.route('/')
    def index():
        return {'message': 'Epitext Backend API', 'version': '1.0.0'}
    
    @app.route('/health')
    def health():
        return {'status': 'healthy'}
    
    # 이미지 서빙 (정적 파일)
    @app.route('/images/<path:filename>')
    def serve_image(filename):
        """이미지 파일 서빙 (원본, 전처리된 이미지, 크롭 이미지)"""
        images_folder = Config.IMAGES_FOLDER
        return send_from_directory(images_folder, filename)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)

