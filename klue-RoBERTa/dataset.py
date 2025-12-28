import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import kss  # pip install kss
from koeda import EDA # pip install koeda konlpy

class VoicePhishingDataset(Dataset):
    """
    보이스피싱 탐지용 데이터셋 클래스
    
    [주요 기능]
    1. 슬라이딩 윈도우: 긴 대화 내용을 5문장씩 묶고, 2문장씩 이동하며 데이터 생성
    2. KSS 문장 분리: 한국어 구어체에 특화된 정교한 문장 분리
    3. KoEDA 증강: 학습 시에만 유의어 교체, 단어 삭제 등을 수행하여 Whisper 인식 오류(STT Error) 대비
    """
    def __init__(self, file_path, tokenizer, max_length=512, window_size=5, stride=2, inference_mode=False):
        """
        Args:
            file_path (str): 분할된 CSV 파일 경로 (예: train_master.csv 또는 test_master.csv)
            tokenizer (PreTrainedTokenizer): RoBERTa 토크나이저
            max_length (int): 토큰 최대 길이 (Default: 512)
            window_size (int): 윈도우 크기 (Default: 5문장)
            stride (int): 이동 간격 (Default: 2문장)
            inference_mode (bool): True일 경우 증강을 끄고 라벨이 없어도 동작함
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        self.inference_mode = inference_mode
        
        # 1. 원본 데이터 로드
        print(f"[Dataset] Loading data from {file_path} (Inference Mode: {inference_mode})...")
        try:
            self.raw_df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"파일을 읽을 수 없습니다: {e}")

        # 2. 증강 모듈 초기화 (학습 모드일 때만 로드하여 메모리 절약)
        if not self.inference_mode:
            print("[Dataset] Initializing KoEDA for text augmentation...")
            try:
                # alpha_sr: 유의어 교체, alpha_ri: 무작위 삽입, alpha_rs: 무작위 교환, alpha_rd: 무작위 삭제
                # Whisper 오류(단어 누락, 오인식) 시뮬레이션을 위해 rd(삭제)와 sr(교체) 비율 설정
                self.eda = EDA(
                    morpheme_analyzer="Okt", 
                    alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1
                )
            except Exception as e:
                print(f"[Warning] KoEDA 초기화 실패 (Augmentation Off): {e}")
                self.eda = None
        else:
            self.eda = None
        
        # 3. 데이터 전처리 (KSS + Sliding Window)
        print(f"[Dataset] Processing with KSS split (Window: {window_size}, Stride: {stride})...")
        self.samples = self._prepare_samples(self.raw_df, inference_mode)
        
        print(f"[Dataset] Ready. Total samples: {len(self.samples)}")

    def _split_sentences(self, text):
        """
        kss 라이브러리를 사용한 정교한 한국어 문장 분리
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        try:
            # kss.split_sentences는 리스트를 반환함
            sentences = kss.split_sentences(text)
            return sentences
        except Exception as e:
            # 안전장치: kss 실패 시 줄바꿈으로 분리
            return text.strip().split('\n')

    def _prepare_samples(self, df, inference_mode):
        processed_data = []
        
        for idx, row in df.iterrows():
            script = row['script']
            
            # 라벨 처리
            if 'label' in row and not pd.isna(row['label']):
                label = int(row['label'])
            else:
                label = 0 if not inference_mode else -1
            
            # 1) 문장 단위로 분리 (KSS)
            sentences = self._split_sentences(script)
            
            if not sentences:
                continue

            # 2) 문장이 윈도우보다 작으면 통째로 하나로 넣음
            if len(sentences) <= self.window_size:
                combined_text = " ".join(sentences)
                processed_data.append({
                    'input_text': combined_text,
                    'label': label
                })
                continue
            
            # 3) 슬라이딩 윈도우 적용 (Window 5, Stride 2)
            #    예: [1,2,3,4,5], [3,4,5,6,7], [5,6,7,8,9] ...
            for i in range(0, len(sentences) - self.window_size + 1, self.stride):
                window = sentences[i : i + self.window_size]
                combined_text = " ".join(window)
                
                processed_data.append({
                    'input_text': combined_text,
                    'label': label
                })
                
        return processed_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        text = data['input_text']
        label = data['label']
        
        # -----------------------------------------------------------
        # [핵심] 텍스트 증강 (Augmentation)
        # -----------------------------------------------------------
        # 학습 모드이고, EDA가 초기화되었으며, 50% 확률에 당첨되면 증강 수행
        if not self.inference_mode and self.eda is not None and random.random() < 0.5:
            try:
                # eda(text)는 증강된 텍스트 리스트를 반환할 수 있음 -> 문자열로 변환
                aug_text = self.eda(text)
                # koeda 버전에 따라 리스트로 반환될 경우 첫 번째 요소 선택
                if isinstance(aug_text, list):
                    text = aug_text[0]
                else:
                    text = aug_text
            except Exception:
                pass # 증강 실패 시 원본 사용

        # -----------------------------------------------------------
        # 토크나이징 (RoBERTa Max Length 512)
        # -----------------------------------------------------------
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }