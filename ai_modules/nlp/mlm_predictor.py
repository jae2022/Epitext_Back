"""
MLM(Masked Language Model) 예측 모듈
BERT 기반 MLM을 사용하여 마스킹된 토큰을 예측합니다.
"""

import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .utils import normalize_mask_tokens


class MLMPredictor:
    """MLM 예측을 담당하는 클래스"""
    
    def __init__(self, config: Dict, device: str = "cpu"):
        """
        MLM 예측기를 초기화합니다.
        
        Args:
            config: 설정 딕셔너리 (nlp_config.json에서 로드)
            device: 연산 디바이스 ('cpu' 또는 'cuda')
        """
        mlm_cfg = config['mlm_model']
        self.model_name = mlm_cfg['model_name']
        self.top_k = mlm_cfg['top_k']
        self.max_length = mlm_cfg['max_length']
        self.device = device
        self.tokenizer = None
        self.model = None
        
    def load_model(self) -> None:
        """모델을 메모리에 로드합니다."""
        print(f"[MLM] 모델 로드 중: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=False
        )
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[MLM] ✓ MLM 모델 로드 완료")
    
    def predict_masks(
        self, 
        text: str
    ) -> List[List[Dict[str, any]]]:
        """
        텍스트 내의 [MASK] 토큰을 예측합니다.
        
        Args:
            text: 마스크가 포함된 텍스트
            
        Returns:
            각 마스크 위치별 top-k 예측 결과 리스트
        """
        # [MASK1], [MASK2] -> [MASK] 정규화
        text_normalized = normalize_mask_tokens(text)
        
        print(f"[MLM] 입력 텍스트 샘플: {text_normalized[:100]}...")
        print(f"[MLM] [MASK] 토큰 개수: {text_normalized.count('[MASK]')}")
        
        # 토크나이즈
        inputs = self.tokenizer(
            text_normalized, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # [MASK] 위치 찾기
        mask_indices = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]
        
        print(f"[MLM] 토크나이저가 찾은 [MASK] 위치 개수: {len(mask_indices)}")
        
        if len(mask_indices) == 0:
            print("[MLM] ⚠️ 경고: [MASK] 토큰을 찾을 수 없습니다!")
            sample_tokens = self.tokenizer.convert_ids_to_tokens(
                inputs['input_ids'][0][:50]
            )
            print(f"[MLM] 토큰화된 입력 샘플: {sample_tokens}")
            return []
        
        # 예측 수행
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # 각 마스크 위치별로 top-k 예측
        all_predictions = []
        for mask_idx in mask_indices:
            mask_logits = logits[0, mask_idx, :]
            
            # 전체 어휘에 대해 softmax 계산 후 top-k 선택
            all_probs = torch.nn.functional.softmax(mask_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(all_probs, self.top_k)
            
            top_k_tokens = self.tokenizer.convert_ids_to_tokens(
                top_k_indices.tolist()
            )
            
            predictions = [
                {
                    "token": token,
                    "probability": float(prob)
                }
                for token, prob in zip(top_k_tokens, top_k_probs.tolist())
            ]
            all_predictions.append(predictions)
        
        return all_predictions

