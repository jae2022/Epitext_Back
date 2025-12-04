# -*- coding: utf-8 -*-
"""
전체 파이프라인 실행 스크립트
전처리 -> OCR -> 구두점 복원 -> SikuRoBERTa -> Swin -> 문맥/획 일치도 계산 -> 번역
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import numpy as np
import logging

# 환경 변수 로드
load_dotenv()
sys.path.insert(0, str(Path('.').absolute()))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_json_dump(data, file_path):
    """numpy 타입을 포함한 데이터를 JSON으로 저장"""
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (tuple, set)):
            return list(obj)
        return obj
    
    converted_data = convert(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

def find_intersection(mlm_top10, swin_top10):
    """MLM과 Swin 예측 결과의 교집합 찾기"""
    mlm_tokens = {item['token']: item['probability'] for item in mlm_top10}
    swin_tokens = {item['token']: item['probability'] for item in swin_top10}
    
    intersection = {}
    for token in mlm_tokens.keys():
        if token in swin_tokens:
            # 교집합 확률 = (MLM 확률 + Swin 확률) / 2
            intersection[token] = (mlm_tokens[token] + swin_tokens[token]) / 2.0
    
    if intersection:
        # 확률이 높은 순으로 정렬
        sorted_intersection = sorted(intersection.items(), key=lambda x: x[1], reverse=True)
        return sorted_intersection[0]  # (token, probability)
    return None

def restore_text_with_predictions(ocr_results, restoration_results):
    """복원 결과를 OCR 결과에 적용하여 텍스트 생성"""
    # 복원 결과를 order로 매핑
    restoration_map = {r.get('order'): r for r in restoration_results}
    
    # OCR 결과를 순서대로 처리하여 텍스트 생성
    restored_chars = []
    for item in ocr_results:
        text = item.get('text', '')
        order = item.get('order')
        item_type = item.get('type', 'TEXT')
        
        # MASK인 경우 복원 결과 사용
        if 'MASK' in item_type and order in restoration_map:
            restored_item = restoration_map[order]
            restored_chars.append(restored_item.get('selected_token', '□'))
        else:
            # 일반 텍스트는 그대로 사용
            restored_chars.append(text)
    
    return ''.join(restored_chars)

def run_full_pipeline(image_path: str, output_dir: str = './test_output'):
    """전체 파이프라인 실행"""
    
    print('=' * 70)
    print('전체 파이프라인 실행')
    print('=' * 70)
    print(f'입력 이미지: {image_path}\n')
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    swin_output = str(output_path / 'swin_output.jpg')
    ocr_output = str(output_path / 'ocr_output.png')
    
    pipeline_result = {
        'preprocessing': {},
        'ocr': {},
        'nlp': {},
        'swin': {},
        'restoration': {},
        'translation': {}
    }
    
    # ========================================================================
    # 1단계: 전처리
    # ========================================================================
    print('[1/5] 전처리 실행 중...')
    try:
        from ai_modules.preprocessor_unified import preprocess_image_unified
        
        preprocess_result = preprocess_image_unified(
            input_path=image_path,
            output_swin_path=swin_output,
            output_ocr_path=ocr_output,
            use_rubbing=True
        )
        
        if not preprocess_result.get('success'):
            raise ValueError(f"전처리 실패: {preprocess_result.get('message')}")
        
        pipeline_result['preprocessing'] = {
            'success': True,
            'swin_path': swin_output,
            'ocr_path': ocr_output,
            'swin_shape': preprocess_result.get('swin', {}).get('output_shape')
        }
        print('✅ 전처리 완료!\n')
        
    except Exception as e:
        logger.error(f"전처리 오류: {e}", exc_info=True)
        pipeline_result['preprocessing'] = {'success': False, 'error': str(e)}
        return pipeline_result
    
    # ========================================================================
    # 2단계: OCR
    # ========================================================================
    print('[2/5] OCR 실행 중...')
    try:
        from ai_modules.ocr_engine import get_ocr_engine
        
        ocr_engine = get_ocr_engine()
        ocr_result = ocr_engine.run_ocr(ocr_output)
        
        if not ocr_result.get('results'):
            raise ValueError("OCR 결과가 비어있습니다.")
        
        # OCR 텍스트 생성
        ocr_text = ''.join([item.get('text', '') for item in ocr_result.get('results', [])])
        
        pipeline_result['ocr'] = {
            'success': True,
            'total_characters': len(ocr_result.get('results', [])),
            'text': ocr_text,
            'results': ocr_result.get('results', [])
        }
        
        mask1_count = sum(1 for r in ocr_result.get('results', []) if 'MASK1' in str(r.get('type', '')))
        mask2_count = sum(1 for r in ocr_result.get('results', []) if 'MASK2' in str(r.get('type', '')))
        print(f'✅ OCR 완료! (총 {len(ocr_result.get("results", []))}개, MASK1: {mask1_count}개, MASK2: {mask2_count}개)\n')
        
    except Exception as e:
        logger.error(f"OCR 오류: {e}", exc_info=True)
        pipeline_result['ocr'] = {'success': False, 'error': str(e)}
        return pipeline_result
    
    # ========================================================================
    # 3단계: 구두점 복원 및 SikuRoBERTa (MLM)
    # ========================================================================
    print('[3/5] 구두점 복원 및 SikuRoBERTa MLM 예측 실행 중...')
    try:
        from ai_modules.nlp_engine import get_nlp_engine
        
        nlp_engine = get_nlp_engine()
        nlp_result = nlp_engine.process_text(
            raw_text=ocr_text,
            ocr_results=ocr_result.get('results', []),
            add_space=True,
            reduce_punc=True
        )
        
        if 'error' in nlp_result:
            raise ValueError(f"NLP 처리 실패: {nlp_result.get('error')}")
        
        pipeline_result['nlp'] = {
            'success': True,
            'punctuated_text': nlp_result.get('punctuated_text_with_masks', ''),
            'results': nlp_result.get('results', []),
            'statistics': nlp_result.get('statistics', {})
        }
        
        print(f'✅ NLP 완료! (구두점 복원 + MLM 예측: {len(nlp_result.get("results", []))}개)\n')
        
    except Exception as e:
        logger.error(f"NLP 오류: {e}", exc_info=True)
        pipeline_result['nlp'] = {'success': False, 'error': str(e)}
        return pipeline_result
    
    # ========================================================================
    # 4단계: Swin MASK2 복원
    # ========================================================================
    print('[4/5] Swin MASK2 복원 실행 중...')
    try:
        from ai_modules.swin_engine import get_swin_engine
        
        swin_engine = get_swin_engine()
        swin_result = swin_engine.run_swin_restoration(swin_output, ocr_result)
        
        if not swin_result.get('results'):
            logger.warning("Swin 복원 결과가 비어있습니다.")
        
        pipeline_result['swin'] = {
            'success': True,
            'results': swin_result.get('results', []),
            'statistics': swin_result.get('statistics', {})
        }
        
        print(f'✅ Swin 완료! (복원된 MASK2: {len(swin_result.get("results", []))}개)\n')
        
    except Exception as e:
        logger.error(f"Swin 오류: {e}", exc_info=True)
        pipeline_result['swin'] = {'success': False, 'error': str(e)}
        return pipeline_result
    
    # ========================================================================
    # 5단계: 문맥 일치도와 획 일치도 계산 및 복원
    # ========================================================================
    print('[5/5] 문맥/획 일치도 계산 및 복원 실행 중...')
    try:
        # NLP 결과와 Swin 결과를 order로 매핑
        nlp_results_map = {r.get('order'): r for r in nlp_result.get('results', [])}
        swin_results_map = {r.get('order'): r for r in swin_result.get('results', [])}
        
        restoration_results = []
        
        # 모든 MASK 처리 (MASK1은 NLP만, MASK2는 NLP + Swin)
        all_masks = set(list(nlp_results_map.keys()) + list(swin_results_map.keys()))
        
        for order in sorted(all_masks):
            nlp_item = nlp_results_map.get(order)
            swin_item = swin_results_map.get(order)
            
            mask_type = 'MASK1'
            if nlp_item:
                mask_type = nlp_item.get('type', 'MASK1')
            elif swin_item:
                mask_type = swin_item.get('type', 'MASK2')
            
            restoration_item = {
                'order': order,
                'type': mask_type,
                'selected_token': '□',
                'selection_method': 'none',
                'context_match': None,
                'stroke_match': None,
                'intersection': None
            }
            
            if mask_type == 'MASK1':
                # MASK1: NLP (문맥 일치도)만 사용
                if nlp_item and nlp_item.get('top_10'):
                    top1 = nlp_item['top_10'][0]
                    restoration_item['selected_token'] = top1['token']
                    restoration_item['selection_method'] = 'context_match_only'
                    restoration_item['context_match'] = {
                        'token': top1['token'],
                        'probability': top1['probability']
                    }
            
            elif mask_type == 'MASK2':
                # MASK2: NLP + Swin (문맥 일치도 + 획 일치도)
                nlp_top10 = nlp_item.get('top_10', []) if nlp_item else []
                swin_top10 = swin_item.get('top_10', []) if swin_item else []
                
                if nlp_top10:
                    restoration_item['context_match'] = {
                        'token': nlp_top10[0]['token'],
                        'probability': nlp_top10[0]['probability'],
                        'top_10': nlp_top10
                    }
                
                if swin_top10:
                    restoration_item['stroke_match'] = {
                        'token': swin_top10[0]['token'],
                        'probability': swin_top10[0]['probability'],
                        'top_10': swin_top10
                    }
                
                # 교집합 찾기
                if nlp_top10 and swin_top10:
                    intersection = find_intersection(nlp_top10, swin_top10)
                    if intersection:
                        restoration_item['intersection'] = {
                            'token': intersection[0],
                            'probability': intersection[1]
                        }
                        restoration_item['selected_token'] = intersection[0]
                        restoration_item['selection_method'] = 'intersection'
                    else:
                        # 교집합이 없으면 문맥 일치도 1등 사용
                        if nlp_top10:
                            restoration_item['selected_token'] = nlp_top10[0]['token']
                            restoration_item['selection_method'] = 'context_match_fallback'
                elif swin_top10:
                    # NLP 결과가 없으면 Swin만 사용
                    restoration_item['selected_token'] = swin_top10[0]['token']
                    restoration_item['selection_method'] = 'stroke_match_only'
                elif nlp_top10:
                    # Swin 결과가 없으면 NLP만 사용
                    restoration_item['selected_token'] = nlp_top10[0]['token']
                    restoration_item['selection_method'] = 'context_match_only'
            
            restoration_results.append(restoration_item)
        
        # 복원된 텍스트 생성
        restored_text = restore_text_with_predictions(ocr_result.get('results', []), restoration_results)
        
        pipeline_result['restoration'] = {
            'success': True,
            'restored_text': restored_text,
            'results': restoration_results
        }
        
        print(f'✅ 복원 완료! (복원된 마스크: {len(restoration_results)}개)\n')
        
    except Exception as e:
        logger.error(f"복원 오류: {e}", exc_info=True)
        pipeline_result['restoration'] = {'success': False, 'error': str(e)}
        return pipeline_result
    
    # ========================================================================
    # 6단계: 번역
    # ========================================================================
    print('[6/6] 번역 실행 중...')
    try:
        from ai_modules.translation_engine import get_translation_engine
        
        translation_engine = get_translation_engine()
        translation_result = translation_engine.translate(restored_text)
        
        if translation_result.get('success'):
            pipeline_result['translation'] = {
                'success': True,
                'reading': translation_result.get('reading', ''),
                'entities': translation_result.get('entities', ''),
                'translation': translation_result.get('translation', ''),
                'model': translation_result.get('model', '')
            }
            print('✅ 번역 완료!\n')
        else:
            pipeline_result['translation'] = {
                'success': False,
                'error': translation_result.get('error', 'Translation failed')
            }
            print(f'⚠️ 번역 실패: {translation_result.get("error")}\n')
        
    except Exception as e:
        logger.error(f"번역 오류: {e}", exc_info=True)
        pipeline_result['translation'] = {'success': False, 'error': str(e)}
    
    # ========================================================================
    # 결과 저장
    # ========================================================================
    output_json = output_path / 'full_pipeline_result.json'
    safe_json_dump(pipeline_result, output_json)
    print(f'💾 전체 파이프라인 결과 저장: {output_json}')
    
    return pipeline_result

if __name__ == '__main__':
    input_image = '/Users/jincerity/Downloads/백시구신도비(白時耉神道碑).png'
    
    if not os.path.exists(input_image):
        print(f'❌ 이미지 파일을 찾을 수 없습니다: {input_image}')
        sys.exit(1)
    
    result = run_full_pipeline(input_image)
    
    # 요약 출력
    print('\n' + '=' * 70)
    print('파이프라인 실행 요약')
    print('=' * 70)
    print(f"전처리: {'✅' if result['preprocessing'].get('success') else '❌'}")
    print(f"OCR: {'✅' if result['ocr'].get('success') else '❌'} ({result['ocr'].get('total_characters', 0)}개)")
    print(f"NLP: {'✅' if result['nlp'].get('success') else '❌'}")
    print(f"Swin: {'✅' if result['swin'].get('success') else '❌'}")
    print(f"복원: {'✅' if result['restoration'].get('success') else '❌'}")
    print(f"번역: {'✅' if result['translation'].get('success') else '❌'}")
    
    if result['translation'].get('success'):
        print(f"\n📝 번역 결과:")
        print(f"  음독: {result['translation'].get('reading', '')[:100]}...")
        print(f"  고유명사: {result['translation'].get('entities', '')[:100]}...")
        print(f"  최종 번역: {result['translation'].get('translation', '')[:200]}...")

