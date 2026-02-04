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
import re
import torch
from refine_transcription import refine_transcription
from faster_whisper import WhisperModel
from tqdm import tqdm

# -------------------- 설정 --------------------
BASE_DIR = "./보이스 피싱 데이터(금감원)"
INPUT_FOLDERS = ["바로 이 목소리", "대출 사기형", "수사기관 사칭형"]
OUTPUT_DIR = os.path.join(BASE_DIR, "text_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZE = "large-v3"
BEAM_SIZE = 5
OVERLAP_SEC = 3
BATCH_SENTENCE_COUNT = 40

# -------------------- Whisper 모델 로드 --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"Using device: {device}, compute_type={compute_type}")

print(f"Loading faster-whisper model '{MODEL_SIZE}'...")
model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
print("Whisper model loaded.")

# -------------------- 메타데이터 구조 --------------------
metadata = {
    "folders": {},
    "total_sentence_count": 0
}

file_global_index = 1  # 저장 파일 번호 1부터 시작

# -------------------- 폴더 순회 --------------------
for folder_name in INPUT_FOLDERS:
    input_path = os.path.join(BASE_DIR, folder_name)
    print(f"\n===== Processing folder: {folder_name} =====")

    # 해당 폴더 내 MP3/WAV 파일 수집
    file_list = [f for f in os.listdir(input_path) if f.lower().endswith((".mp3", ".wav"))]

    # 숫자 추출 후 정렬
    file_dict = {}
    for f in file_list:
        m = re.findall(r'\d+', f)
        if m:
            num = int(m[0])
            if num in file_dict:
                # 중복 시 접미사 붙이기
                suffix = 1
                while f"{num}_{suffix}" in file_dict:
                    suffix += 1
                file_dict[f"{num}_{suffix}"] = f
            else:
                file_dict[num] = f

    # 숫자 순 정렬
    sorted_keys = sorted(file_dict.keys(), key=lambda x: int(str(x).split("_")[0]))

    folder_start_index = file_global_index
    folder_sentence_count = 0
    converted_count = 0

    for key in tqdm(sorted_keys, desc=f"Files in {folder_name}", ncols=100):
        filename = file_dict[key]
        in_file = os.path.join(input_path, filename)
        out_filename = f"{file_global_index}.txt"
        out_file = os.path.join(OUTPUT_DIR, out_filename)

        # -------------------- 이미 변환된 파일 존재하면 건너뛰기 + sentence_count 복원 --------------------
        if os.path.exists(out_file):
            print(f"[SKIP] File #{file_global_index} exists → Restoring metadata from {out_filename}")
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    restored_sentence_count = sum(1 for _ in f)
            except:
                restored_sentence_count = 0
            folder_sentence_count += restored_sentence_count
            metadata["total_sentence_count"] += restored_sentence_count
            converted_count += 1
            file_global_index += 1
            continue

        # -------------------- 변환 실행 --------------------
        print(f"Processing file #{file_global_index}: {filename}")
        sentence_count = refine_transcription(
            input_file=in_file,
            output_file=out_file,
            model=model,
            beam_size=BEAM_SIZE,
            overlap_sec=OVERLAP_SEC,
            batch_sentence_count=BATCH_SENTENCE_COUNT
        )

        folder_sentence_count += sentence_count
        metadata["total_sentence_count"] += sentence_count
        converted_count += 1
        file_global_index += 1

    folder_end_index = file_global_index - 1

    # -------------------- 폴더 메타데이터 기록 --------------------
    metadata["folders"][folder_name] = {
        "start": folder_start_index,
        "end": folder_end_index,
        "count": converted_count,
        "sentence_count": folder_sentence_count
    }

# -------------------- 전체 메타데이터 저장 --------------------
meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\n===== All processing done =====")
print(f"Metadata saved at: {meta_path}")
