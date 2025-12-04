# -*- coding: utf-8 -*-
"""
독립 실행 가능한 OCR 스크립트
Google Vision API + HRCenterNet 앙상블 기반 한자 OCR 및 손상 영역 탐지

사용법:
    python dong_ocr.py <이미지_경로>
    
예시:
    python dong_ocr.py test_image.png
    python dong_ocr.py /path/to/image.jpg
"""

import os
import sys
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger("DONG_OCR")

# OCR 엔진 및 전처리 모듈 import
try:
    from ai_modules.ocr_engine import get_ocr_engine
    from ai_modules.preprocessor_unified import preprocess_image_unified
except ImportError as e:
    logger.error(f"❌ 모듈 import 실패: {e}")
    logger.error("ai_modules 폴더와 필요한 모델 파일들이 있는지 확인하세요.")
    sys.exit(1)


def format_ocr_results(raw_results, image_filename):
    """
    OCR 결과를 요청하신 JSON 포맷으로 변환하는 함수
    
    Args:
        raw_results: OCR 엔진에서 반환된 results 리스트
        image_filename: 이미지 파일명
        
    Returns:
        포맷팅된 결과 딕셔너리
    """
    formatted_list = []
    
    if not raw_results:
        logger.warning("⚠️ 원본 OCR 결과가 비어있습니다.")
        return {
            "image": image_filename,
            "results": []
        }
    
    order_counter = 0
    for idx, item in enumerate(raw_results):
        if not isinstance(item, dict):
            logger.warning(f"⚠️ 잘못된 데이터 형식 (idx={idx}): {type(item)}")
            continue
        
        # 좌표 추출 및 리스트 변환 (여러 키 이름 지원)
        min_x = item.get('min_x')
        min_y = item.get('min_y')
        max_x = item.get('max_x')
        max_y = item.get('max_y')
        
        # 대체 키 확인
        if min_x is None:
            x_val = item.get('x', 0)
            min_x = x_val
        if min_y is None:
            y_val = item.get('y', 0)
            min_y = y_val
        if max_x is None:
            max_x = item.get('x2')
            if max_x is None:
                x_val = item.get('x', 0)
                width = item.get('width', 0)
                max_x = x_val + width if width > 0 else 0
        if max_y is None:
            max_y = item.get('y2')
            if max_y is None:
                y_val = item.get('y', 0)
                height = item.get('height', 0)
                max_y = y_val + height if height > 0 else 0
        
        # float 변환
        try:
            min_x = float(min_x) if min_x is not None else 0.0
            min_y = float(min_y) if min_y is not None else 0.0
            max_x = float(max_x) if max_x is not None else 0.0
            max_y = float(max_y) if max_y is not None else 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ 좌표 변환 실패 (idx={idx}): {e}")
            continue
        
        # 좌표가 모두 0이고 width/height도 없으면 스킵
        if min_x == 0 and min_y == 0 and max_x == 0 and max_y == 0:
            width = item.get('width', 0)
            height = item.get('height', 0)
            if width > 0 and height > 0:
                # center_x, center_y로 재구성 시도
                center_x = item.get('center_x', width / 2)
                center_y = item.get('center_y', height / 2)
                min_x = float(center_x - width / 2)
                min_y = float(center_y - height / 2)
                max_x = float(center_x + width / 2)
                max_y = float(center_y + height / 2)
            else:
                logger.warning(f"⚠️ 좌표가 모두 0이고 width/height도 없음 (idx={idx}, text={item.get('text', '')}) - 스킵")
                continue
        
        # 유효성 검사
        if max_x <= min_x or max_y <= min_y:
            logger.warning(f"⚠️ 잘못된 좌표 범위 (idx={idx}): ({min_x}, {min_y}) -> ({max_x}, {max_y}) - 스킵")
            continue
        
        new_item = {
            "order": order_counter,  # 0부터 시작하는 연속된 순서
            "text": item.get('text', ''),
            "type": item.get('type', 'TEXT'),
            "box": [min_x, min_y, max_x, max_y],  # 요청하신 좌표 포맷
            "confidence": float(item.get('confidence', 0.0)),
            "source": item.get('source', 'Unknown')
        }
        formatted_list.append(new_item)
        order_counter += 1
    
    return {
        "image": image_filename,
        "results": formatted_list
    }


def draw_bboxes(image_path, results, output_path):
    """
    이미지에 색상별 Bounding Box를 그리고 저장하는 함수
    
    색상 구분:
    - 🟢 초록색: Google OCR
    - 🟣 보라색: Custom OCR
    - 🔵 파란색: MASK1 (짙은 먹물)
    - 🔴 빨간색: MASK2 (부분 오염)
    
    Args:
        image_path: 원본 이미지 경로
        results: OCR 결과 리스트 (포맷팅된 형식)
        output_path: 출력 이미지 경로
    """
    try:
        # 이미지 로드 (한글 경로 지원)
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            # 한글 경로가 안 되면 일반 방법 시도
            img = cv2.imread(image_path)
            if img is None:
                logger.warning("❌ 이미지 로드 실패 (시각화 건너뜀)")
                return
        
        box_count = 0
        
        # 색상 정의 (OpenCV는 BGR 순서)
        COLOR_GREEN = (0, 255, 0)      # Google OCR
        COLOR_PURPLE = (255, 0, 255)   # Custom OCR
        COLOR_BLUE = (255, 0, 0)       # MASK1
        COLOR_RED = (0, 0, 255)         # MASK2
        COLOR_YELLOW = (0, 255, 255)   # 기타 (Inferred, GapFill 등)
        
        for item in results:
            # 'box' 키에서 좌표 가져오기
            box = item.get('box', [0, 0, 0, 0])
            if not isinstance(box, list) or len(box) != 4:
                logger.warning(f"⚠️ 잘못된 box 형식: {box} (order={item.get('order', 'unknown')})")
                continue
                
            try:
                x1 = int(float(box[0]))
                y1 = int(float(box[1]))
                x2 = int(float(box[2]))
                y2 = int(float(box[3]))
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"⚠️ 좌표 변환 실패: {box} (order={item.get('order', 'unknown')}) - {e}")
                continue
            
            # 좌표 유효성 검사
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"⚠️ 잘못된 좌표 범위: ({x1}, {y1}) -> ({x2}, {y2}) (order={item.get('order', 'unknown')})")
                continue
            
            # 이미지 크기 범위 확인 및 조정
            img_h, img_w = img.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(x1 + 1, min(x2, img_w))
                y2 = max(y1 + 1, min(y2, img_h))
            
            # 정보 가져오기
            text = item.get('text', '')
            source = item.get('source', '')
            item_type = item.get('type', 'TEXT')
            
            # 색상 결정 로직
            if '[MASK1]' in text or item_type == 'MASK1':
                color = COLOR_BLUE
            elif '[MASK2]' in text or item_type == 'MASK2':
                color = COLOR_RED
            elif source == 'Google':
                color = COLOR_GREEN
            elif source == 'Custom':
                color = COLOR_PURPLE
            else:
                color = COLOR_YELLOW
            
            # 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 텍스트 표시 (선택사항 - 작은 글자는 생략)
            if item_type == 'TEXT' and len(text) <= 2:
                # 한 글자만 표시
                cv2.putText(
                    img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
            elif 'MASK' in item_type:
                # MASK 타입 표시
                label = '[M1]' if item_type == 'MASK1' else '[M2]'
                cv2.putText(
                    img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
            
            box_count += 1
        
        # 결과 저장 (한글 경로 지원)
        extension = os.path.splitext(output_path)[1].lower()
        if extension in ['.jpg', '.jpeg']:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        elif extension == '.png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            encode_param = []
        
        result, encoded_img = cv2.imencode(extension, img, encode_param)
        if result:
            with open(output_path, mode='wb') as f:
                encoded_img.tofile(f)
            logger.info(f"🖼️  B-Box 이미지 저장됨: {output_path} ({box_count}개 박스)")
            logger.info(f"   (🟢Google, 🟣Custom, 🔵MASK1, 🔴MASK2)")
        else:
            logger.error("❌ 이미지 인코딩 실패")
            
    except Exception as e:
        logger.error(f"❌ 시각화 중 오류 발생: {e}", exc_info=True)


def run_ocr(image_path, use_preprocessing=True):
    """
    OCR 실행 및 결과 저장
    
    Args:
        image_path: 입력 이미지 경로
        use_preprocessing: 전처리 사용 여부 (기본: True)
        
    Returns:
        성공 여부
    """
    if not os.path.exists(image_path):
        logger.error(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return False
    
    logger.info(f"🚀 OCR 분석 시작: {image_path}")
    
    try:
        # 전처리 (선택사항)
        ocr_image_path = image_path
        preprocess_result = {'success': False}  # 기본값 설정
        if use_preprocessing:
            logger.info("📸 이미지 전처리 중...")
            base_dir = os.path.dirname(os.path.abspath(image_path))
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 전처리 결과 저장 경로
            swin_path = os.path.join(base_dir, f"{base_name}_swin_temp.jpg")
            ocr_preprocessed_path = os.path.join(base_dir, f"{base_name}_ocr_temp.png")
            
            preprocess_result = preprocess_image_unified(
                input_path=image_path,
                output_swin_path=swin_path,
                output_ocr_path=ocr_preprocessed_path,
                use_rubbing=True
            )
            
            if preprocess_result.get('success'):
                ocr_image_path = ocr_preprocessed_path
                logger.info(f"✅ 전처리 완료: {ocr_preprocessed_path}")
            else:
                logger.warning(f"⚠️ 전처리 실패, 원본 이미지 사용: {preprocess_result.get('message')}")
                ocr_image_path = image_path
        
        # OCR 엔진 로드
        engine = get_ocr_engine()
        logger.info("✅ OCR 엔진 로드 완료")
        
        # OCR 실행 (전처리된 이미지 사용)
        try:
            raw_result = engine.run_ocr(ocr_image_path)
        except Exception as ocr_exception:
            logger.error(f"❌ OCR 실행 중 예외 발생: {ocr_exception}", exc_info=True)
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False
        
        if not raw_result:
            logger.error("❌ OCR 결과가 None입니다.")
            return False
        
        if not isinstance(raw_result, dict):
            logger.error(f"❌ OCR 결과가 딕셔너리가 아닙니다: {type(raw_result)}")
            return False
        
        if not raw_result.get('success'):
            error_msg = raw_result.get('error', 'Unknown Error')
            logger.error(f"❌ OCR 실패: {error_msg}")
            logger.error(f"   raw_result: {raw_result}")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("✅ OCR 분석 완료")
        logger.info(f"  - Google 인식: {raw_result.get('google_count', 0)}개")
        logger.info(f"  - Custom 인식: {raw_result.get('custom_count', 0)}개")
        logger.info(f"  - 최종 결과: {raw_result.get('final_count', 0)}개")
        logger.info(f"  - 텍스트 줄 수: {raw_result.get('columns', 0)}")
        logger.info("-" * 60)
        
        # 텍스트 라인 출력
        for i, line in enumerate(raw_result.get('text_lines', []), 1):
            logger.info(f"  [열 {i}] {line}")
        
        logger.info("="*60)
        
        # 원본 결과 확인
        raw_results = raw_result.get('results', [])
        if not raw_results:
            logger.error("❌ OCR 결과가 비어있습니다!")
            logger.error(f"   - raw_result keys: {list(raw_result.keys())}")
            logger.error(f"   - final_count: {raw_result.get('final_count', 0)}")
            return False
        
        logger.info(f"[DEBUG] 원본 OCR 결과 개수: {len(raw_results)}")
        if len(raw_results) > 0:
            logger.info(f"[DEBUG] 첫 번째 결과 샘플 키: {list(raw_results[0].keys())}")
        
        # 결과 데이터 포맷 변환
        image_filename = os.path.basename(image_path)
        formatted_result = format_ocr_results(raw_results, image_filename)
        
        formatted_results = formatted_result.get('results', [])
        logger.info(f"[DEBUG] 포맷팅 후 결과 개수: {len(formatted_results)}")
        
        if not formatted_results:
            logger.error("❌ 포맷팅된 결과가 비어있습니다!")
            logger.error(f"   - 원본 결과 개수: {len(raw_results)}")
            if raw_results:
                logger.error(f"   - 첫 번째 원본 항목: {raw_results[0]}")
            return False
        
        # JSON 저장
        json_path = os.path.splitext(image_path)[0] + "_ocr_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_result, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 JSON 결과 저장됨: {json_path}")
        
        # B-Box 이미지 생성
        # 전처리된 이미지를 사용한 경우, 전처리된 이미지에 bbox를 그려야 좌표가 맞음
        output_img_path = os.path.splitext(image_path)[0] + "_bbox.jpg"
        bbox_image_path = ocr_image_path if use_preprocessing and preprocess_result.get('success') else image_path
        draw_bboxes(bbox_image_path, formatted_result['results'], output_img_path)
        
        # 통계 출력
        results = formatted_result['results']
        google_count = sum(1 for r in results if r['source'] == 'Google')
        custom_count = sum(1 for r in results if r['source'] == 'Custom')
        mask1_count = sum(1 for r in results if r['type'] == 'MASK1')
        mask2_count = sum(1 for r in results if r['type'] == 'MASK2')
        text_count = sum(1 for r in results if r['type'] == 'TEXT')
        
        logger.info("\n" + "="*60)
        logger.info("📊 최종 통계")
        logger.info(f"  - 🟢 Google: {google_count}개")
        logger.info(f"  - 🟣 Custom: {custom_count}개")
        logger.info(f"  - 🔵 MASK1 (짙은 먹물): {mask1_count}개")
        logger.info(f"  - 🔴 MASK2 (부분 오염): {mask2_count}개")
        logger.info(f"  - 📝 TEXT: {text_count}개")
        logger.info("="*60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OCR 실행 중 오류 발생: {e}", exc_info=True)
        return False


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("="*60)
        print("📖 사용법:")
        print("  python dong_ocr.py <이미지_경로>")
        print("")
        print("예시:")
        print("  python dong_ocr.py test_image.png")
        print("  python dong_ocr.py /path/to/image.jpg")
        print("="*60)
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # 환경 변수 확인
    if not os.getenv('OCR_WEIGHTS_BASE_PATH'):
        logger.error("❌ OCR_WEIGHTS_BASE_PATH 환경 변수가 설정되지 않았습니다.")
        logger.error("   .env 파일에 OCR_WEIGHTS_BASE_PATH를 설정하세요.")
        sys.exit(1)
    
    if not os.getenv('GOOGLE_CREDENTIALS_JSON'):
        logger.error("❌ GOOGLE_CREDENTIALS_JSON 환경 변수가 설정되지 않았습니다.")
        logger.error("   .env 파일에 GOOGLE_CREDENTIALS_JSON을 설정하세요.")
        sys.exit(1)
    
    # OCR 실행
    success = run_ocr(image_path)
    
    if success:
        logger.info("✅ 모든 작업 완료!")
        sys.exit(0)
    else:
        logger.error("❌ 작업 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()

