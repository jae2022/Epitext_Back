"""
이미지 처리 유틸리티 함수
"""
import os
from PIL import Image


def crop_rubbing_image(image_path, crop_x, crop_y, crop_width, crop_height, output_path):
    """
    탁본 이미지에서 특정 영역을 크롭하여 저장
    
    Args:
        image_path: 원본 이미지 경로
        crop_x: 크롭 영역 X 좌표
        crop_y: 크롭 영역 Y 좌표
        crop_width: 크롭 영역 너비
        crop_height: 크롭 영역 높이
        output_path: 저장할 경로
        
    Returns:
        저장된 이미지 경로
    """
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 이미지 열기
        img = Image.open(image_path)
        
        # 크롭 영역 계산
        left = crop_x
        top = crop_y
        right = crop_x + crop_width
        bottom = crop_y + crop_height
        
        # 이미지 크롭
        cropped_img = img.crop((left, top, right, bottom))
        
        # 저장
        cropped_img.save(output_path, quality=95)
        
        return output_path
    except Exception as e:
        raise Exception(f"이미지 크롭 실패: {str(e)}")


def save_uploaded_image(file, upload_folder, filename):
    """
    업로드된 이미지 파일 저장
    
    Args:
        file: Flask request.files의 파일 객체
        upload_folder: 저장할 폴더 경로
        filename: 저장할 파일명
        
    Returns:
        저장된 파일 경로
    """
    try:
        # 업로드 폴더 생성
        os.makedirs(upload_folder, exist_ok=True)
        
        # 파일 경로
        file_path = os.path.join(upload_folder, filename)
        
        # 파일 저장
        file.save(file_path)
        
        return file_path
    except Exception as e:
        raise Exception(f"이미지 저장 실패: {str(e)}")

