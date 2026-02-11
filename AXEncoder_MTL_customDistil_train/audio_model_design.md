# Audio Phishing Detection Model Design (Final Implementation)

본 문서는 `AXEncoder_MTL_customDistil_train` 폴더에 구현된 **계층적(Hierarchical) 보이스 피싱 탐지 모델**의 상세 설계 및 구현 내용을 설명합니다.

---

## 1. 개요 (Overview)

*   **목표**: 오디오 데이터(통화 내용)를 분석하여 피싱 여부(Phishing vs Normal)를 실시간/배치로 판별.
*   **핵심 접근법**: 
    1.  **STT (Speech-to-Text)**: 고성능 모델(`faster-whisper-large-v3-turbo-ct2`)을 사용하여 오디오를 텍스트로 변환.
    2.  **계층적 모델링 (Hierarchical Modeling)**: 긴 대화의 문맥을 파악하기 위해 **[청크 인코더] + [문맥 통합기]** 구조 채택.
    3.  **효율성**: `bfloat16` 정밀도 및 Frozen Encoder 사용으로 학습 속도 최적화.

---

## 2. 데이터 전처리 파이프라인 (Preprocessing Pipeline)

### 2.1 STT 변환 (`preprocessing/stt_transcriber.py`)
*   **모델**: `deepdml/faster-whisper-large-v3-turbo-ct2`
    *   선정 이유: Large-v3의 성능을 유지하면서 속도가 대폭 개선된 Turbo 모델.
*   **설정**:
    *   **Device**: CUDA (GPU)
    *   **Precision**: `bfloat16` (메모리 절약 및 속도 향상, `float16` 대비 안정성 우수)
    *   **Language**: Korean (`ko`)
*   **입력**: `data/preprocessing/카테고리/파일명/청크.wav` (Flat 구조)
*   **출력**: `data/preprocessing/카테고리/파일명_transcription.json`
    *   JSON 포맷: `[{"start": 0.0, "end": 4.5, "text": "여보세요..."}, ...]`

### 2.2 데이터셋 생성 (`preprocessing/create_dataset.py`)
*   **기능**: 생성된 JSON 파일들을 읽어 학습용 CSV 파일로 변환.
*   **Labeling**:
    *   **Phishing (1)**: `대출 사기형`, `수사기관 사칭형`, `바로 이 목소리`
    *   **Normal (0)**: `이체 출금 대출서비스`, `잔고 및 거래내역`, `일반 대화`
*   **출력 파일**:
    *   `data/train.csv`: 학습 데이터 (전체 90%)
    *   `data/val.csv`: 검증 데이터 (전체 10%, Stratified Split 적용)

---

## 3. 모델 아키텍처 (Model Architecture)

긴 대화(Long Sequence)를 처리하기 위해 **Hierarchical Structure**를 사용합니다.

### 3.1 텍스트 인코더 (Chunk Encoder)
*   **역할**: 개별 발화(Chunk)의 텍스트를 고차원 특징 벡터(Embedding)로 변환.
*   **모델**: `skt/A.X-Encoder-base` (가정) 또는 호환되는 한국어 Pretrained LLM.
*   **동작**:
    *   입력: 텍스트 청크 (Tokenized)
    *   출력: `[CLS]` Token Embedding (768-dim)
*   **최적화**: **Frozen** 상태로 사용하여 학습 시 역전파(Backprop)를 수행하지 않음.

#### 💡 설계 의도 (Why?)
*   **긴 시퀀스 처리**: 보이스피싱 통화는 수십 분 이상 지속될 수 있어, 전체 텍스트를 한 번에 BERT/Transformer에 넣으면 토큰 길이 제한(512/1024)을 초과합니다.
*   **효율성**: 전체 모델을 Fine-tuning하는 대신, 강력한 Pretrained 언어 모델을 "특징 추출기(Feature Extractor)"로만 사용하여 GPU 메모리 사용량을 최소화했습니다.

---

### 3.2 문맥 통합기 (Context Aggregator)
*   **역할**: 연속된 청크 임베딩들의 흐름(Context)을 파악하여 피싱 징후 포착.
*   **모델**: **Bi-LSTM (Bidirectional LSTM)**
*   **Pooling**: **Max Pooling** (시퀀스 내 가장 강력한 피싱 신호 추출)

#### 💡 설계 의도 (Why?)
*   **문맥 파악**: 피싱범은 초반에 신뢰를 쌓고 중후반에 금전을 요구하는 패턴을 보입니다. Bi-LSTM은 대화의 **순차적 흐름**과 **전후 문맥**을 모두 고려할 수 있습니다.
*   **희소 신호 포착**: 피싱 발언(예: "계좌 이체", "검찰청")은 긴 대화 중 짧게 등장할 수 있습니다. Attention이나 Max Pooling은 이러한 **Critical Moment**를 놓치지 않도록 돕습니다.

#### ✅ 장점 (Pros)
1.  **긴 문맥 처리 가능**: 수천 토큰 이상의 대화도 청크 단위로 처리하여 문맥 유지가 가능함.
2.  **학습/추론 속도**: Frozen Encoder 사용으로 학습 파라미터 수가 적어 가볍고 빠름.
3.  **데이터 효율성**: 적은 양의 피싱 데이터로도 과적합(Overfitting) 없이 학습 가능 (Pretrained 지식 활용).

#### ⚠️ 단점 및 한계 (Cons)
1.  **청크 분절**: 문장이 청크 경계에서 잘릴 경우 의미 파악이 어려울 수 있음 (Overlay로 완화 가능).
2.  **Sequential 속도**: LSTM의 순차적 특성으로 인해 Transformer 기반 모델보다 병렬 처리가 느릴 수 있음.
3.  **미세조정 제약**: Encoder가 고정되어 있어, "피싱 특화 어휘"에 대한 임베딩 조정이 불가능함.

---

### 3.3 분류기 (Classifier Head)
*   **구조**: `Linear(512) -> ReLU -> Dropout(0.3) -> Linear(1)`
*   **출력**: 0~1 사이의 확률값 (Sigmoid 적용 전 Logit).

---

## 4. 학습 전략 (Training Strategy)

*   **Loss Function**: `BCEWithLogitsLoss` (이진 분류 표준)
*   **Optimizer**: `AdamW` (Learning Rate: 2e-5 ~ 1e-4)
*   **Batch Size**: 32 (GPU 메모리에 맞춰 조정)
*   **Metrics**:
    *   **F1-Score**: 불균형 데이터(Imbalanced Data)에서 정확도보다 중요한 지표.
    *   **Accuracy**: 보조 지표.

---

## 5. 프로젝트 폴더 구조 (Project Structure)

### 5.1 코드 구조 (`AXEncoder_MTL_customDistil_train/`)
```text
AXEncoder_MTL_customDistil_train/
├── analysis/                   # [New] 오디오 분석 및 전처리 (Raw -> Chunk)
│   ├── analyze_audio_stats.py  # 오디오 통계 분석 (dB 계산)
│   ├── process_audio.py        # 노이즈 제거 및 침묵 기반 자르기
│   └── requirements.txt        # 분석용 라이브러리 목록
├── data/                       # 생성된 학습 데이터 (CSV)
│   ├── train.csv               # 학습용 데이터셋
│   └── val.csv                 # 검증용 데이터셋
├── model/                      # 모델 정의
│   ├── ax_encoder.py           # Text Encoder Wrapper
│   └── hierarchical_model.py   # 전체 모델 아키텍처
├── preprocessing/              # 전처리 스크립트
│   ├── stt_transcriber.py      # Audio -> Text (JSON) 변환
│   └── create_dataset.py       # JSON -> CSV 변환
├── train.py                    # 학습 실행 스크립트
├── inference.py                # 단일 파일 추론 스크립트
├── dataset.py                  # PyTorch Dataset & Collator
├── config.py                   # 설정 변수 모음
├── utils.py                    # 유틸리티 (로깅, 시드 고정)
└── requirements.txt            # 필요 라이브러리 목록
```

### 5.2 데이터 폴더 구조 (Data Storage)
`config.py`의 `DATA_ROOT`가 가리키는 실제 데이터 저장소입니다. (기본값: 프로젝트 상위 폴더 `../data`)
```text
data/
└── preprocessing/              # DATA_ROOT (자동 생성됨)
    ├── 대출 사기형/             # 카테고리 폴더
    │   ├── conv123_chunk_0.wav
    │   ├── conv123_chunk_1.wav
    │   └── conv123_transcription.json  # STT 결과
    ├── 수사기관 사칭형/
    │   └── ...
    └── 일반 대화/
        └── ...
```

---

## 6. 사용 방법 (Usage)

### Step 1. 환경 설정
필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
pip install -r analysis/requirements.txt
```
*   `ffmpeg`가 설치되어 있어야 합니다 (pydub 의존성).

### Step 2. 데이터 준비 및 생성 (Data Generation)

학습에 필요한 데이터(`data/preprocessing/...`)를 생성하는 과정입니다.

#### 1. 원본 데이터 준비
프로젝트 폴더 외부에 원본 오디오 파일(`raw_data`)을 준비하는 것을 권장합니다.
```text
/home/j2hoon10/raw_data/
├── 대출 사기형/
│   ├── voice1.wav
│   └── ...
├── 수사기관 사칭형/
└── 일반 대화/
```

#### 2. 오디오 통계 분석 (Optional)
오디오의 평균 볼륨(dBFS)을 분석하여, 노이즈 제거 및 침묵 제거 시 사용할 적절한 임계값을 확인합니다.
```bash
python analysis/analyze_audio_stats.py "/home/j2hoon10/raw_data/대출 사기형"
# 결과로 audio_stats.json 생성됨 (추천 임계값 확인 가능)
```

#### 3. 오디오 전처리 (Raw -> Chunk)
원본 오디오에서 노이즈를 제거하고, 침묵 구간을 기준으로 잘라서 `data/preprocessing`에 저장합니다.
```bash
# 사용법: python analysis/process_audio.py [원본폴더경로] --gpu
# 예시:
python analysis/process_audio.py "/home/j2hoon10/raw_data/대출 사기형" --gpu
python analysis/process_audio.py "/home/j2hoon10/raw_data/수사기관 사칭형" --gpu
python analysis/process_audio.py "/home/j2hoon10/raw_data/일반 대화" --gpu
```
*   **실행 결과**: `../data/preprocessing/대출 사기형/` 폴더에 `_chunk_0000.wav` 파일들이 생성됩니다.

### Step 3. STT 변환 및 데이터셋 생성
생성된 오디오 청크를 텍스트로 변환하고 학습용 CSV를 생성합니다.

```bash
# 1. 오디오 -> 텍스트 변환 (JSON 생성)
# 결과: data/preprocessing/{카테고리}/{파일명}_transcription.json
python preprocessing/stt_transcriber.py

# 2. JSON -> CSV 변환 (학습 데이터 생성)
# 결과: data/train.csv, data/val.csv
python preprocessing/create_dataset.py
```

### Step 4. 모델 학습
```bash
python train.py
```
*   학습된 모델은 `model/` 폴더에 `best_model.pt`로 저장됩니다.

### Step 5. 추론 (Inference)
```bash
python inference.py "path/to/test_audio.wav"
```
