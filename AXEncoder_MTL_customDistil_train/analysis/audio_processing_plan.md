# 음성 데이터 전처리 및 분할 시스템 (Audio Processing System)

이 문서는 구현된 음성 데이터 전처리 시스템의 상세 로직, 사용법 및 내부 알고리즘을 설명합니다. 이 시스템은 **2단계(Two-Stage) 파이프라인**으로 구성되어 전체 데이터셋의 통계적 일관성을 유지하며 데이터를 처리합니다.

## 1. 시스템 개요 (Overview)

본 시스템은 대량의 오디오 파일을 분석하여 **배경 소음을 제거**하고, **침묵 구간을 기반으로 문장 단위로 분할**하는 기능을 제공합니다.

### 주요 특징
*   **2단계 처리**: 전체 데이터셋 분석(Stage 1) 후 일관된 임계값으로 처리(Stage 2)하여 파일 간 품질 격차 해소.
*   **GPU 가속**: `torch` 및 CUDA/MPS를 활용한 고속 노이즈 제거 지원.
*   **적응형 노이즈 제거**: Spectral Gating 알고리즘을 통해 목소리는 살리고 배경 잡음만 제거.
*   **정밀한 분할**: 글로벌 통계 기반의 고정 임계값(Fixed Threshold)과 파일별 동적 임계값(Dynamic Threshold) 모드 지원.

---

## 2. 처리 파이프라인 (Processing Pipeline)

### Stage 1: 데이터셋 통계 분석 (`analyze_audio_stats.py`)
전체 오디오 파일을 스캔하여 볼륨(dBFS) 분포를 분석하고, 최적의 **침묵 임계값(Silence Threshold)**을 산출합니다.

*   **입력**: 오디오 파일 디렉토리
*   **로직**:
    1.  모든 파일의 dBFS(평균 데시벨) 측정.
    2.  전체 평균(Global Mean) 및 표준편차 산출.
    3.  `Global Mean + Offset` 공식으로 권장 침묵 임계값 계산.
*   **출력**: `audio_stats.json` (통계 및 권장 설정값 저장)

### Stage 2: 노이즈 제거 및 분할 (`process_audio.py`)
Stage 1에서 생성된 파라미터 파일을 로드하여 실제 오디오 처리를 수행합니다.

*   **입력**: 오디오 파일 디렉토리, `audio_stats.json` (선택)
*   **로직**:
    1.  **Noise Reduction**: `noisereduce` 라이브러리의 Spectral Gating 알고리즘 적용 (GPU 지원).
    2.  **Silence Segmentation**: JSON에서 로드한 글로벌 임계값(또는 파일별 상대값)을 기준으로 침묵 구간 탐지.
    3.  **Export**: 분할된 오디오 청크를 WAV 파일로 저장.
*   **출력**: 처리된 오디오 파일들 (`chunk_0000.wav`, ...)

---

## 3. 상세 알고리즘 (Algorithm Details)

### 3.1. 노이즈 제거 (Noise Reduction)
*   **라이브러리**: `noisereduce`, `torch`
*   **알고리즘**: Non-stationary Noise Reduction (Spectral Gating)
*   **파라미터**:
    *   `--strength` (0.0 ~ 1.0): 노이즈 제거 강도. 값이 클수록 더 많은 노이즈를 제거하지만 목소리 왜곡 가능성이 있음. (기본값: 0.8)
    *   `--gpu`: 활성화 시 CUDA(NVIDIA) 또는 MPS(Apple Silicon) 가속 사용.

### 3.2. 침묵 구간 분할 (Segmentation)
*   **라이브러리**: `pydub`, `numpy`
*   **임계값 결정 방식**:
    *   **Mode A (권장): Global Fixed Threshold**
        *   `--params_file`로 JSON을 로드하여 데이터셋 전체에 동일한 dB 기준 적용.
        *   데이터 품질이 균일할 때 유리.
    *   **Mode B: Dynamic Threshold**
        *   JSON이 없을 경우, 각 파일의 평균 dBFS 기준으로 상대적 임계값(`Mean + Offset`) 적용.
        *   녹음 환경이 파일마다 제각각일 때 유리.
*   **파라미터**:
    *   `--min_silence` (ms): 이 시간 이상 침묵이 지속되어야 자름 (기본값: 700ms).
    *   `--thresh_offset` (dB): 평균 볼륨 대비 침묵으로 간주할 레벨 차이 (기본값: -16dB).

---

## 4. 사용 가이드 (Usage Guide)

### 1단계: 통계 분석 (Analysis)
```bash
python analysis/analyze_audio_stats.py "C:\Path\To\Input" --output_json "analysis/audio_stats.json" --thresh_offset -16
```
*   `--thresh_offset`: 평균 dB보다 얼마나 낮아야 침묵으로 볼 것인지 설정 (예: -16이면 평균 -20dB인 파일에서 -36dB 이하를 침묵으로 간주).

### 2단계: 변환 실행 (Processing)
```bash
python analysis/process_audio.py "C:\Path\To\Input" --params_file "analysis/audio_stats.json" --gpu
```
*   `--params_file`: 1단계에서 생성한 JSON 파일 경로.
*   `--gpu`: GPU 가속 사용 (가능한 경우).
*   `--strength`: 노이즈 제거 강도 조절 (예: 0.9).
*   `--min_silence`: 최소 침묵 길이 (예: 500).

---

## 5. 파일 구조 (File Structure)

| 파일명 | 설명 |
|---|---|
| `analyze_audio_stats.py` | (Stage 1) 전체 오디오 통계 분석 및 파라미터 추출 |
| `process_audio.py` | (Stage 2) 노이즈 제거 및 파일 분할 실행 스크립트 |
| `requirements.txt` | 필요 라이브러리 목록 (`noisereduce`, `pydub`, `torch` 등) |
