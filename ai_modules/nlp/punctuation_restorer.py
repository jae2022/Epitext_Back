"""
구두점 복원 모듈
Hugging Face 모델을 사용하여 한국어 고전 텍스트의 구두점을 복원합니다.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from huggingface_hub import snapshot_download
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class PunctuationRestorer:
    """구두점 복원을 담당하는 클래스"""
    
    def __init__(self, config: Dict, cache_dir: str, device: str = "cpu"):
        """
        구두점 복원기를 초기화합니다.
        
        Args:
            config: 설정 딕셔너리 (nlp_config.json에서 로드)
            cache_dir: 모델 캐시 디렉토리 (기본 경로)
            device: 연산 디바이스 ('cpu' 또는 'cuda')
        """
        punc_cfg = config['punc_model']
        self.model_tag = punc_cfg['model_tag']
        self.max_length = punc_cfg['max_length']
        self.window_size = punc_cfg['window_size']
        self.overlap = punc_cfg['overlap']
        
        self.cache_dir = Path(cache_dir) / "punc"
        self.device = device
        self.model_info = None
        
    def download_model(self) -> None:
        """Hugging Face에서 모델을 다운로드합니다."""
        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.cache_dir.exists() or not any(self.cache_dir.iterdir()):
            print(f"[PUNC] 모델 다운로드 중: {self.model_tag}")
            snapshot_download(
                repo_id=self.model_tag,
                repo_type="model",
                local_dir=str(self.cache_dir),
                local_dir_use_symlinks=False,
            )
        else:
            print(f"[PUNC] 캐시된 모델 사용: {self.cache_dir}")
    
    def load_model(self) -> None:
        """모델을 메모리에 로드합니다."""
        torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        
        # 모델 파일 찾기
        fnames = sorted(self.cache_dir.rglob("*.safetensors"))
        if len(fnames) == 0:
            # safetensors가 없으면 다른 형식 시도
            fnames = sorted(self.cache_dir.rglob("*.bin"))
        
        if len(fnames) == 0:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.cache_dir}")
        
        hface_path = fnames[0].parent
        
        # 토크나이저 및 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(
            str(hface_path), 
            model_max_length=self.max_length
        )
        model = AutoModelForTokenClassification.from_pretrained(
            str(hface_path), 
            device_map=self.device if "cuda" in self.device else None, 
            torch_dtype=torch_dtype
        )
        if "cuda" not in self.device:
            model = model.to(self.device)
        model.eval()
        
        # NER 파이프라인 생성
        ner_pipeline = pipeline(
            task="ner", 
            model=model, 
            tokenizer=tokenizer,
            device=0 if "cuda" in self.device else -1
        )
        
        # 레이블 매핑 로드
        label2id_path = hface_path / "label2id.json"
        if not label2id_path.is_file():
            label2id_path = hface_path.parent / "label2id.json"
        if not label2id_path.is_file():
            raise FileNotFoundError(f"label2id.json을 찾을 수 없습니다: {hface_path}")
        
        label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
        
        self.model_info = {
            "model": model,
            "tokenizer": tokenizer,
            "pipe": ner_pipeline,
            "label2id": label2id
        }
        
        print(f"[PUNC] ✓ 구두점 복원 모델 로드 완료")
    
    def restore_punctuation(
        self,
        text: str,
        add_space: bool = True,
        reduce: bool = True,
    ) -> str:
        """
        슬라이딩 윈도우 방식으로 구두점을 복원합니다.
        
        Args:
            text: 입력 텍스트
            add_space: 구두점 뒤 공백 추가 여부
            reduce: 구두점 단순화 여부
            
        Returns:
            구두점이 복원된 텍스트
        """
        if not text.strip():
            return ""
        
        # 레이블 -> 구두점 매핑 생성
        label2punc = self._build_label2punc(add_space, reduce)
        
        # 슬라이딩 윈도우로 레이블 예측
        labels = self._predict_labels_sliding(text, self.window_size, self.overlap)
        
        # 길이 조정
        if len(labels) < len(text):
            labels += ["O"] * (len(text) - len(labels))
        elif len(labels) > len(text):
            labels = labels[:len(text)]
        
        # 구두점 삽입
        result = ""
        for ch, label in zip(text, labels):
            result += ch
            punc = label2punc.get(label, "")
            result += punc
        
        return result.strip()
    
    def _predict_labels_sliding(
        self, 
        text: str, 
        window_size: int, 
        overlap: int
    ) -> List[str]:
        """
        슬라이딩 윈도우로 각 문자의 레이블을 예측합니다.
        
        Args:
            text: 입력 텍스트
            window_size: 윈도우 크기
            overlap: 중첩 크기
            
        Returns:
            각 문자에 대한 레이블 리스트
        """
        n = len(text)
        if n == 0:
            return []
        
        # 각 위치별 후보 레이블 저장
        labels_per_pos = [[] for _ in range(n)]
        stride = max(1, window_size - overlap)
        start = 0
        
        while start < n:
            end = min(start + window_size, n)
            sub_text = text[start:end]
            
            try:
                # NER 예측 수행
                sub_preds = self.model_info["pipe"](sub_text)
                _, sub_labels = self._align_predictions(sub_text, sub_preds)
            except Exception as e:
                # 오류 발생 시 모두 'O' 레이블
                print(f"[PUNC] 예측 오류 (start={start}): {e}")
                sub_labels = ["O"] * len(sub_text)
            
            # 전역 위치에 레이블 저장
            for i, label in enumerate(sub_labels):
                gidx = start + i
                if gidx >= n:
                    break
                if label != "O":
                    labels_per_pos[gidx].append(label)
            
            if end == n:
                break
            start += stride
        
        # 다수결 투표로 최종 레이블 결정
        final_labels = []
        for cand_list in labels_per_pos:
            if not cand_list:
                final_labels.append("O")
            else:
                c = Counter(cand_list)
                label, _ = c.most_common(1)[0]
                final_labels.append(label)
        
        return final_labels
    
    @staticmethod
    def _align_predictions(text: str, predictions: List[dict]) -> Tuple[List[str], List[str]]:
        """
        NER 예측 결과를 문자 단위 레이블로 정렬합니다.
        
        Args:
            text: 원본 텍스트
            predictions: NER 예측 결과
            
        Returns:
            (문자 리스트, 레이블 리스트) 튜플
        """
        words = list(text)
        labels = ["O" for _ in range(len(words))]
        
        for pred in predictions:
            idx = pred["end"] - 1
            if 0 <= idx < len(labels):
                labels[idx] = pred["entity"]
        
        return words, labels
    
    def _build_label2punc(self, add_space: bool, reduce: bool) -> Dict[str, str]:
        """
        레이블을 구두점으로 매핑하는 딕셔너리를 생성합니다.
        
        Args:
            add_space: 구두점 뒤 공백 추가 여부
            reduce: 구두점 단순화 여부
            
        Returns:
            레이블 -> 구두점 매핑 딕셔너리
        """
        label2id = self.model_info["label2id"]
        label2punc = {f"B-{v}": k for k, v in label2id.items()}
        label2punc["O"] = ""
        
        # 구두점 단순화
        if reduce:
            new_label2punc = {}
            for label, punc in label2punc.items():
                if label == "O":
                    new_label2punc[label] = ""
                else:
                    reduced = self._reduce_punc(punc)
                    new_label2punc[label] = reduced
            label2punc = new_label2punc
        
        # 공백 추가
        if add_space:
            special_puncs = "!,:;?。"
            label2punc = {
                k: self._insert_space(v, special_puncs) 
                for k, v in label2punc.items()
            }
            label2punc["O"] = ""
        
        return label2punc
    
    @staticmethod
    def _reduce_punc(text: str) -> str:
        """
        구두점을 단순화합니다 (?, 。, , 중 하나로 변환).
        
        Args:
            text: 구두점 문자열
            
        Returns:
            단순화된 구두점
        """
        reduce_map = {
            ",": ",", "-": ",", "/": ",", ":": ",", "|": ",", 
            "·": ",", "、": ",",
            "?": "?", "!": "。", ".": "。", ";": "。", "。": "。",
        }
        
        text = "".join([reduce_map.get(c, "") for c in text])
        punc_order = "?。,,"
        
        if len(set(text).intersection(punc_order)) == 0:
            return ""
        
        # 가장 많이 등장한 구두점 선택
        counts = {c: text.count(c) for c in punc_order}
        max_count = max(counts.values())
        max_keys = {k for k, v in counts.items() if v == max_count}
        
        if len(max_keys) == 1:
            return max_keys.pop()
        
        # 동률일 경우 우선순위에 따라 선택
        for c in punc_order:
            if c in max_keys:
                return c
        
        return ""
    
    @staticmethod
    def _insert_space(text: str, chars: str) -> str:
        """
        특정 문자 뒤에 공백을 삽입합니다.
        
        Args:
            text: 원본 텍스트
            chars: 공백을 추가할 문자들
            
        Returns:
            공백이 삽입된 텍스트
        """
        result = ""
        for c in text:
            result += c
            if c in chars:
                result += " "
        return result

