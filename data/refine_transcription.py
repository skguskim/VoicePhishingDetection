import os
from pydub.utils import which
from kospellpy import spell_init
from tqdm import tqdm

# -------------------- 전역 함수 --------------------
def correct_text_batch(spell_checker, text_batch):
    """하나의 배치(문장 합친 문자열)에 대한 맞춤법 교정"""
    try:
        corrected = spell_checker(text_batch)
        sentences = [s.strip() for s in corrected.split(". ") if s.strip()]
        return sentences
    except:
        return text_batch.split(". ")

def remove_duplicates_and_short(sentences, min_len=3):
    """
    한국어 특화 휴리스틱
    - 중복 문장 제거
    - 너무 짧은 문장 제거
    """
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        cleaned = sentence.rstrip(".!?").strip()
        if len(cleaned) < min_len:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique_sentences.append(sentence)
    return unique_sentences

# -------------------- 메인 처리 함수 --------------------
def refine_transcription(
    input_file,
    output_file,
    model,
    beam_size=5,
    overlap_sec=2,
    batch_sentence_count=20
):
    """
    - 배치 단위 맞춤법 검사
    - 한국어 특화 중복 제거 + 짧은 문장 제거 (맞춤법 전)
    - tqdm 진행바를 통한 실시간 진행률 표시
    """
    # --- 0. ffmpeg 확인 ---
    if which("ffmpeg") is None:
        print("Warning: ffmpeg not found. Audio conversion may fail.")

    # --- 1. 음성 인식 + VAD 기반 분할 ---
    print(f"Transcribing audio: {input_file}")
    segments, _ = model.transcribe(
        input_file,
        beam_size=beam_size,
        language="ko",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    # --- 2. 의미 있는 문장 추출 + 오버랩 처리 (진행률 표시) ---
    all_sentences = []
    last_end_time = 0
    print("Extracting meaningful sentences...")
    for seg in tqdm(segments, desc="Processing segments", ncols=100):
        seg_start = max(seg.start - overlap_sec, last_end_time)
        seg_end = seg.end
        text = seg.text.strip()
        if text and len(text) > 8 and any(c.isalnum() for c in text):
            all_sentences.append(text)
        last_end_time = seg_end

    print(f"Total meaningful segments before filtering: {len(all_sentences)}")

    # --- 3. 중복 제거 + 짧은 문장 제거 (맞춤법 전) ---
    all_sentences = remove_duplicates_and_short(all_sentences, min_len=3)
    print(f"After deduplication & length filter: {len(all_sentences)} sentences")

    # --- 4. 맞춤법 교정 ---
    print("Applying Korean spell check...")
    spell_checker = spell_init()
    refined_sentences = []

    for i in tqdm(range(0, len(all_sentences), batch_sentence_count), desc="Spellchecking batches", ncols=100):
        batch_text = " ".join(all_sentences[i:i+batch_sentence_count])
        refined_sentences.extend(correct_text_batch(spell_checker, batch_text))

    # --- 5. 최종 TXT 파일로 저장 ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in refined_sentences:
            f.write(sentence + "\n")

    print(f"Saved refined text to: {output_file}")
    return len(refined_sentences)
