import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# ==========================================
# 설정 (Configuration)
# ==========================================
# 모델 ID 
MODEL_ID = "answerdotai/ModernBERT-base"

# 데이터 경로 
DATA_PATH = "data/dataset_master.csv"

# 하이퍼파라미터
MAX_LEN = 1024   # 메모리 부족 시 512로 줄이기 (ModernBERT는 최대 8192 지원)
BATCH_SIZE = 8   # GPU 메모리에 따라 4 ~ 16 조절
EPOCHS = 3
LEARNING_RATE = 5e-5

# ==========================================
# 데이터 로드 및 전처리
# ==========================================
def load_and_preprocess_data(path):
    print(f"📂 Loading data from {path}...")
    
    # 1. CSV 읽기 (인코딩 에러 발생 시 encoding='cp949' 또는 'euc-kr' 시도)
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='cp949')

    print(f"   - 전체 데이터 개수: {len(df)}")

    # 2. 컬럼 이름 변경 (script -> text)
    if 'script' in df.columns:
        df = df.rename(columns={'script': 'text'})
        print("   - 컬럼 이름 변경 완료: 'script' -> 'text'")
    
    # 필수 컬럼 확인
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"데이터셋에 필수 컬럼이 없습니다. 현재 컬럼: {df.columns}")

    # 3. 결측치 제거
    df = df.dropna(subset=['text', 'label'])
    
    # 4. 라벨 타입 정수형으로 변환 (혹시 모를 에러 방지)
    df['label'] = df['label'].astype(int)

    return df

# 데이터 로드
df = load_and_preprocess_data(DATA_PATH)

# ==========================================
# 층화 추출 (Stratified Split) - 불균형 데이터 필수
# ==========================================
# 피싱(1)과 정상(0) 비율을 유지하면서 Train/Validation 나누기
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label'] # 핵심: 라벨 비율 유지
)

print(f"\n📊 데이터 분할 결과:")
print(f"   - Train set: {len(train_df)} (Phishing: {sum(train_df['label']==1)})")
print(f"   - Val set  : {len(val_df)} (Phishing: {sum(val_df['label']==1)})")

# HuggingFace Dataset으로 변환
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ==========================================
# 토크나이저 및 모델 로드
# ==========================================
print("\n🚀 Loading ModernBERT Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 전처리 함수
def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=MAX_LEN, 
        padding=False # Dynamic padding을 위해 여기서는 False
    )

# 토큰화 적용
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# 모델 로드 (0: 정상, 1: 피싱 -> 2개의 라벨)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    num_labels=2,
    trust_remote_code=True
)

# ==========================================
# 학습 설정 (Trainer)
# ==========================================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",    # 매 epoch마다 평가
    save_strategy="epoch",    # 매 epoch마다 저장
    load_best_model_at_end=True, # 가장 성능 좋은 모델 불러오기
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(), # GPU 사용 시 가속
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"
)

# 배치 처리 시 패딩을 동적으로 맞춰주는 Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ==========================================
#  학습 시작
# ==========================================
print("\n🔥 Starting Training...")
trainer.train()

# ==========================================
# 최종 모델 저장
# ==========================================
SAVE_PATH = "./final_modernbert_model"
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n✅ Training Finished! Model saved to {SAVE_PATH}")