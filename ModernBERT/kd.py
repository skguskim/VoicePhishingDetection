import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 설정 (Configuration)
# ==========================================
class Config:
    TEACHER_PATH = "./final_modernbert_model"  # 학습시킨 모델 경로
    BASE_MODEL_ID = "answer/ModernBERT-base" # Student 모델 베이스
    DATA_PATH = "data/dataset_master.csv"       # 데이터 경로 
    
    # 지식 증류 하이퍼파라미터 
    ALPHA = 0.5      # Hard Loss와 Soft Loss의 비율 (보통 0.5 or 0.3)
    TEMPERATURE = 4.0 # Softmax를 부드럽게 만드는 온도 (T)
    
    # 학습 설정
    MAX_LEN = 1024
    BATCH_SIZE = 8
    EPOCHS = 5      # 학생은 선생님보다 좀 더 오래 배우는 게 좋음
    LEARNING_RATE = 5e-5
    SEED = 42

set_seed(Config.SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 커스텀 Trainer (지식 증류 로직 핵심)
# ==========================================
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # 선생님은 학습하지 않으므로 평가 모드로 설정 & 그래디언트 차단
        self.teacher.eval()
        self.teacher.to(self.args.device)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # student의 출력 계산
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # teacher의 출력 계산 (Gradient 계산 안 함)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # 실제 정답(Labels) 가져오기
        labels = inputs.get("labels")

        # Loss 계산 
        # Hard Loss: 학생이 정답을 맞췄는가? (CrossEntropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Soft Loss: 학생이 선생님의 확률 분포를 닮았는가? (KLDiv)
        # T^2를 곱해주는 것은 Softmax의 그라디언트 스케일을 보정하기 위함
        soft_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / Config.TEMPERATURE, dim=1),
            F.softmax(teacher_logits / Config.TEMPERATURE, dim=1)
        ) * (Config.TEMPERATURE ** 2)

        # 최종 Loss 결합
        loss = (Config.ALPHA * hard_loss) + ((1 - Config.ALPHA) * soft_loss)

        return (loss, student_outputs) if return_outputs else loss

# ==========================================
# 데이터 로드 
# ==========================================
def load_data():
    df = pd.read_csv(Config.DATA_PATH)
    if 'script' in df.columns: df = df.rename(columns={'script':'text'})
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df

# 데이터 준비
df = load_data()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=Config.SEED, stratify=df['label'])
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_PATH)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=Config.MAX_LEN, padding=False)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# ==========================================
# 모델 준비 (Teacher & Student)
# ==========================================
print(f"👨‍🏫 Teacher 모델 로드 중: {Config.TEACHER_PATH}")
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    Config.TEACHER_PATH,
    num_labels=2
)

print("👶 Student 모델 생성 중 (레이어 축소)...")
# 선생님 설정 가져오기
student_config = AutoConfig.from_pretrained(Config.BASE_MODEL_ID, num_labels=2)

# 레이어 개수를 반으로 줄이기 (22 -> 11)
original_layers = student_config.num_hidden_layers # # 원래 레이어 수
student_config.num_hidden_layers = original_layers // 2 # 반절로 나누기
print(f"   - Teacher Layers: {original_layers} -> Student Layers: {student_config.num_hidden_layers}")

# 학생 모델 초기화
# 아예 랜덤 초기화보다는, Pretrained된 앞부분 레이어를 가져오는게 학습이 빠름
# ignore_mismatched_sizes=True를 쓰면 레이어 개수가 달라도 호환되는 부분(임베딩 등)은 가져옴
student_model = AutoModelForSequenceClassification.from_pretrained(
    Config.BASE_MODEL_ID,
    config=student_config,
    ignore_mismatched_sizes=True 
)

# ==========================================
# 학습 시작 (Distillation)
# ==========================================
training_args = TrainingArguments(
    output_dir="./distilled_results",
    learning_rate=Config.LEARNING_RATE,
    per_device_train_batch_size=Config.BATCH_SIZE,
    per_device_eval_batch_size=Config.BATCH_SIZE,
    num_train_epochs=Config.EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=True, # Loss 기준 가장 좋은 모델 로드
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 커스텀 Trainer 사용
trainer = DistillationTrainer(
    teacher_model=teacher_model, # 선생님 모델 전달
    model=student_model,         # 학습할 학생 모델
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\n🔥 Knowledge Distillation Start...")
trainer.train()

# ==========================================
# 저장
# ==========================================
SAVE_PATH = "./final_student_model"
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n✅ Student Model saved to {SAVE_PATH}")