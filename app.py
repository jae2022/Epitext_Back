"""
Flask 애플리케이션 진입점
"""
from flask import Flask
from flask_cors import CORS
from config import Config
from models import db
from routes.rubbings import rubbings_bp
from routes.targets import targets_bp
from routes.inspection import inspection_bp
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
    
    # 필요한 디렉토리 생성
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.IMAGES_FOLDER, exist_ok=True)
    os.makedirs(Config.CROPPED_IMAGES_FOLDER, exist_ok=True)
    
    @app.route('/')
    def index():
        return {'message': 'Epitext Backend API', 'version': '1.0.0'}
    
    @app.route('/health')
    def health():
        return {'status': 'healthy'}
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)

