# 지식 증류 기반 텍스트 보이스피싱 탐지 시스템
### Knowledge Distillation-based Text Voice Phishing Detection System

서울과학기술대학교 인공지능응용학과  
김나현, 정지훈, 장수효

---

## Overview

보이스피싱은 특정 키워드만으로 판단하기 어렵고, 대화가 진행됨에 따라 개인정보 요구, 계좌 이체 유도, 기관 사칭 등 다양한 위험 신호가 점진적으로 누적되는 특성을 가진다.

본 프로젝트는 통화 음성을 STT(Speech-to-Text)로 변환한 뒤, Transformer 기반 자연어 처리 모델을 이용하여 보이스피싱 여부를 탐지하는 텍스트 기반 탐지 시스템을 제안한다.

또한 Teacher-Student 기반 지식 증류(Knowledge Distillation)를 적용하여 높은 성능을 유지하면서도 경량화된 Student 모델을 구축하였으며, 이를 FastAPI 기반 실시간 데모 시스템에 적용하였다.

---

## Features

- Whisper 기반 음성 → 텍스트 변환
- 보이스피싱 / 정상 대화 분류
- KLUE-RoBERTa 기반 Teacher 모델 학습
- Knowledge Distillation 기반 Student 모델 학습
- ModernBERT 기반 추가 실험
- FastAPI 기반 실시간 추론 서비스
- STT 오류를 고려한 텍스트 데이터 처리

---

## System Pipeline

```text
Audio Input
    ↓
Whisper STT
    ↓
Text Preprocessing
    ↓
Teacher Model Training
    ↓
Knowledge Distillation
    ↓
Student Model
    ↓
Real-time Inference
    ↓
Voice Phishing Prediction
```

---

## Repository Structure

```text
VoicePhishingDetection/
│
├── data/
│   ├── KorCCVi.csv
│   ├── dataset_master.csv
│   ├── role_insertion.py
│   ├── role_deletion.py
│   ├── refine_transcription.py
│   └── run_data_preprocessing.py
│
├── klue-RoBERTa/
│   ├── architecture.py
│   ├── dataset.py
│   ├── loss_fun.py
│   ├── train_teacher.py
│   ├── train_student.py
│   └── trainer.py
│
├── ModernBERT/
│   ├── train.py
│   └── kd.py
│
├── app2/
│   ├── main.py
│   ├── architecture.py
│   ├── config.py
│   └── static/
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Dataset

본 프로젝트는 일반 대화와 보이스피싱 대화 데이터를 이용하여 학습 데이터를 구성하였다.

### Data Sources

#### Normal Conversation

- AI Hub 일반 대화 데이터
- AI Hub 상담 데이터

#### Voice Phishing Conversation

- 금융감독원 보이스피싱 음성 녹취 데이터

---

## Data Preprocessing

전처리 과정에서는 다음과 같은 작업을 수행한다.

### role_insertion.py

화자 태그가 제거된 데이터를 복원한다.

### role_deletion.py

특정 화자의 발화를 제거하여 데이터셋을 정제한다.

### refine_transcription.py

STT 결과의 불필요한 텍스트를 정리한다.

### run_data_preprocessing.py

전체 데이터 전처리 파이프라인을 실행한다.

---

## Teacher Model

Teacher 모델은 KLUE-RoBERTa를 기반으로 구성된다.

```text
Input Text
    ↓
KLUE-RoBERTa Encoder
    ↓
Classification Head
    ↓
Prediction
```

Teacher 모델은 높은 분류 성능을 확보하는 역할을 수행하며, 이후 Student 모델 학습에 사용된다.

---

## Knowledge Distillation

본 프로젝트는 Teacher 모델의 지식을 Student 모델로 전달하기 위해 Knowledge Distillation을 적용한다.

### Distillation Objective

Student 모델은 다음 정보를 학습한다.

- Ground Truth Label
- Teacher Prediction Distribution
- Teacher Hidden Representation

### Loss Function

```text
L_total =
α · CrossEntropy Loss
+ β · Knowledge Distillation Loss
+ γ · Cosine Similarity Loss
```

| Loss | Description |
|---|---|
| CrossEntropy Loss | 실제 정답 라벨 기반 분류 손실 |
| KD Loss | Teacher와 Student의 확률 분포 정렬 |
| Cosine Loss | Hidden State 표현 정렬 |

---

## ModernBERT Experiments

추가적으로 ModernBERT 기반 Teacher 및 Student 실험을 수행하였다.

### Training

```bash
python ModernBERT/train.py
```

### Knowledge Distillation

```bash
python ModernBERT/kd.py
```

---

## Real-time Demo System

FastAPI 기반 실시간 데모 시스템을 제공한다.

### Inference Pipeline

```text
Audio Upload
    ↓
Whisper STT
    ↓
Text Extraction
    ↓
Student Model Inference
    ↓
Prediction Score
```

### Run Demo

```bash
uvicorn app2.main:app --reload
```

기본 주소:

```text
http://localhost:8000
```

---

## Training

### Teacher Model

```bash
python klue-RoBERTa/train_teacher.py
```

### Student Model

```bash
python klue-RoBERTa/train_student.py
```

---

## Environment Setup

### Conda

```bash
conda env create -f environment.yml
conda activate capstone
```

### Pip

```bash
pip install -r requirements.txt
```

---

## Key Contributions

### 1. Text-based Voice Phishing Detection

음성 통화를 텍스트로 변환한 뒤 자연어 처리 기반으로 보이스피싱 여부를 탐지한다.

### 2. Knowledge Distillation

Teacher 모델의 성능을 유지하면서 경량 Student 모델을 구축하였다.

### 3. Real-time Deployment

FastAPI 기반 실시간 추론 환경을 구현하였다.

### 4. STT-aware Pipeline

실제 음성 환경에서 발생하는 STT 오류를 고려한 데이터 처리 파이프라인을 설계하였다.

---

## Data Policy

### Notice

본 저장소에는 실제 음성 데이터 및 개인정보가 포함된 데이터셋이 포함되어 있지 않다.

다음 항목은 Git 저장소에 업로드하지 않는다.

- 원본 음성 파일
- 전사 결과 파일
- 개인정보 포함 데이터
- 학습 체크포인트
- 학습 로그

---

## References

1. Liu et al., RoBERTa: A Robustly Optimized BERT Pretraining Approach, 2019.
2. Gu and Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023.
3. Pappagari et al., Hierarchical Transformers for Long Document Classification, 2019.
4. Sim et al., Voice Phishing Detection Scheme using a GPT-3.5 based Large Language Model, 2024.
5. Park et al., Enhanced Voice Phishing Detection using an LLM-based Framework for Data Augmentation and Classification, IEEE Access, 2025.
