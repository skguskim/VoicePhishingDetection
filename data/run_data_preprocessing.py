"""
==========================================================
환경 및 설치 안내 (Python 3.10 기준)
==========================================================

1️⃣ Python 가상환경 생성 (권장)
> conda create -n capstone python=3.10
> conda activate capstone

2️⃣ 필요한 Python 라이브러리 설치
> pip install faster-whisper pydub numpy tqdm kospellpy
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

3️⃣ ffmpeg 설치 (Windows)
> choco install ffmpeg
> ffmpeg -version

==========================================================
💡 모델 파라미터 조절 안내 (faster-whisper)
==========================================================

1. 모델 사이즈 (MODEL_SIZE)
- "tiny", "base", "small", "medium", "large-v2", "large-v3" 등 선택 가능
- 크기가 클수록 정확도 ↑, 속도 ↓, VRAM ↑
- 예: MODEL_SIZE="large-v3"

2. Beam search 크기 (BEAM_SIZE)
- 디코딩 시 탐색 깊이를 조절
- 값이 클수록 정확도 ↑, 속도 ↓
- 예: BEAM_SIZE=5

3. 오버랩 시간 (OVERLAP_SEC)
- VAD 기반 분할 시 인접 구간 중첩 시간
- 음성 구간 잘림 방지용
- 예: OVERLAP_SEC=2 (초)

4. 맞춤법 검사 배치 크기 (BATCH_SENTENCE_COUNT)
- 한 번에 맞춤법 검사할 문장 수
- 값이 클수록 처리 효율 ↑, 메모리 사용 ↑
- 예: BATCH_SENTENCE_COUNT=20

==========================================================
💡 실행 예시
==========================================================

from refine_transcription import refine_transcription
from faster_whisper import WhisperModel
import torch
import os

# --- 모델 한 번만 로드 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel("large-v3", device=device, compute_type=compute_type)

# --- 단일 파일 처리 ---
INPUT_MP3_FILE = "./수사기관 사칭형(검찰, 경찰 등)/63.mp3"
OUTPUT_FILE = os.path.splitext(INPUT_MP3_FILE)[0] + "_refined.txt"

sentence_count = refine_transcription(
    input_file=INPUT_MP3_FILE,
    output_file=OUTPUT_FILE,
    model=model,
    beam_size=5,          # 필요 시 조정
    overlap_sec=2,        # 필요 시 조정
    batch_sentence_count=20  # 필요 시 조정
)

print(f"총 문장 수: {sentence_count}")
==========================================================
💡 폴더 내 모든 MP3 처리 예시
==========================================================

# process_mp3_files.py 참고
# - INPUT_DIR 폴더 내 모든 MP3 파일을 숫자 순으로 처리
# - 중복 제거 + 짧은 문장 제거 + 맞춤법 교정
# - output TXT 파일과 metadata.json 생성

==========================================================
"""


import os
import json
import torch
from refine_transcription import refine_transcription
from faster_whisper import WhisperModel

# -------------------- 설정 --------------------
INPUT_DIR = "./수사기관 사칭형(검찰, 경찰 등)"
OUTPUT_DIR = "./대화 데이터"
MODEL_SIZE = "large-v3"
BEAM_SIZE = 5
OVERLAP_SEC = 3
BATCH_SENTENCE_COUNT = 40

# -------------------- Whisper 모델 로드 (한 번만) --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"Using device: {device}, compute_type={compute_type}")
print(f"Loading faster-whisper model '{MODEL_SIZE}'...")
model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
print("Whisper model loaded.")

# -------------------- MP3 파일 숫자 순 정렬 --------------------
file_list = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp3")],
    key=lambda x: int(os.path.splitext(x)[0])
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
META_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
all_meta = {}
total_sentences = 0

# -------------------- 반복 처리 --------------------
for filename in file_list:
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = os.path.splitext(filename)[0] + "_refined.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\nProcessing file: {filename}")
    sentence_count = refine_transcription(
        input_file=input_path,
        output_file=output_path,
        model=model,
        beam_size=BEAM_SIZE,
        overlap_sec=OVERLAP_SEC,
        batch_sentence_count=BATCH_SENTENCE_COUNT
    )

    all_meta[filename] = {
        "sentence_count": sentence_count,
        "output_file": output_path
    }
    total_sentences += sentence_count

all_meta["total_sentence_count"] = total_sentences

# -------------------- 메타데이터 저장 --------------------
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(all_meta, f, ensure_ascii=False, indent=2)

print(f"\nAll files processed. Metadata saved to: {META_FILE}")
