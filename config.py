import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Flask 애플리케이션 설정"""
    
    # 데이터베이스 설정
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'epitext_db')
    
    # SQLAlchemy 설정
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        if DB_PASSWORD
        else f"sqlite:///{DB_NAME}.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Flask 설정
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # 파일 업로드 설정
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB
    
    # 이미지 저장 경로
    IMAGES_FOLDER = os.getenv('IMAGES_FOLDER', './images/rubbings')
    CROPPED_IMAGES_FOLDER = os.getenv('CROPPED_IMAGES_FOLDER', './images/rubbings/cropped')

