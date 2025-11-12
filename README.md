# VoicePhishingDetection

보이스피싱 탐지/분석을 위한 데이터 전처리 및 모델 파이프라인(초안).

## 🔧 Dataset
- 입력: `KorCCVi.csv`  
- 필수 컬럼: `transcript` (콜 전체 대화 텍스트)  
- 선택 컬럼: `call_id` (없으면 자동 생성)

## 🎯 Preprocessing Goals
1) **사기범 문장만 추출** → `scammer_only_text` (콜 단위)
2) **피싱 수법 분류** → `primary_method` (콜 단위, 키워드 매칭 다수결/우선순위)

### 수법 카테고리(초안)
- 기관사칭, 원격제어앱, 대환대출/저금리, 검사비/보증금/세금, 상품권/가상자산,
  결제/피싱링크, 협박/압박, 택배/관세/환불

> 사전은 `scripts/preprocess.py` 내 정규식으로 정의. 데이터 특성에 맞춰 지속적으로 보정하세요.

## 📁 Outputs
- `KorCCVi_sentences.csv` (문장 단위)
  - `call_id, sent_id, speaker, is_scammer, phishing_method, text`
- `KorCCVi_aug.csv` (콜 단위 확장)
  - `scammer_only_text, primary_method, methods_detected, scammer_sentence_count`

## ▶️ Run
```bash
python scripts/preprocess.py