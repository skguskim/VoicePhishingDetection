import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset

# ==========================================
# 설정 (Configuration)
# ==========================================
class Config:
    MODEL_ID = "answerdotai/ModernBERT-base"
    DATA_PATH = "data/dataset_master.csv" # 경로 확인 필요
    SAVE_PATH = "./final_modernbert_model"
    
    # 하이퍼파라미터
    MAX_LEN = 1024
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    SEED = 42
    
    # 데이터 로더 설정 (속도 향상)
    NUM_WORKERS = 2 

# 재현성을 위한 시드 고정
set_seed(Config.SEED)

# ==========================================
# 유틸리티 함수
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_and_preprocess_data(path):
    print(f"📂 Loading data from {path}...")
    
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='cp949')

    print(f"   - 전체 데이터 개수: {len(df)}")

    # 컬럼 이름 변경 (script -> text)
    if 'script' in df.columns:
        df = df.rename(columns={'script': 'text'})
    
    # 필수 컬럼 확인
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"데이터셋에 필수 컬럼이 없습니다. 현재 컬럼: {df.columns}")

    # 결측치 제거
    df = df.dropna(subset=['text', 'label'])
    
    # 라벨 타입 정수형으로 변환
    df['label'] = df['label'].astype(int)

    return df

# ==========================================
# 실행 로직
# ==========================================

# 1. 데이터 로드 (Config.DATA_PATH 사용)
df = load_and_preprocess_data(Config.DATA_PATH)

# 2. 층화 추출
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=Config.SEED, 
    stratify=df['label']
)

print(f"\n📊 데이터 분할 결과:")
print(f"   - Train set: {len(train_df)} (Phishing: {sum(train_df['label']==1)})")
print(f"   - Val set  : {len(val_df)} (Phishing: {sum(val_df['label']==1)})")

# Dataset 변환
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 3. 토크나이저 및 모델 로드 (Config 변수 사용)
print("\n🚀 Loading ModernBERT Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)

def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=Config.MAX_LEN, 
        padding=False 
    )

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    Config.MODEL_ID, 
    num_labels=2,
    trust_remote_code=True
)

# 4. 학습 설정 (Config 변수 사용 및 오타 수정)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=Config.LEARNING_RATE,
    per_device_train_batch_size=Config.BATCH_SIZE,
    per_device_eval_batch_size=Config.BATCH_SIZE,
    num_train_epochs=Config.EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # 중복 제거: F1을 기준으로 설정
    fp16=torch.cuda.is_available(),
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",                # 콤마 추가됨 (중요!)
    dataloader_num_workers=Config.NUM_WORKERS,
    save_total_limit=2,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. 학습 시작
print("\n🔥 Starting Training...")
trainer.train()

# 6. 저장
trainer.save_model(Config.SAVE_PATH)
tokenizer.save_pretrained(Config.SAVE_PATH)
print(f"\n✅ Training Finished! Model saved to {Config.SAVE_PATH}")