import os
from pydub.utils import which
from kospellpy import spell_init
from tqdm import tqdm
import re

# -------------------- 전역 함수 --------------------
def correct_text_batch(spell_checker, text_batch):
    """하나의 배치(문장 합친 문자열)에 대한 맞춤법 교정"""
    try:
        corrected = spell_checker(text_batch)
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', corrected) if s.strip()]
        return sentences
    except:
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text_batch)]

def remove_duplicates_and_short(sentences, min_len=3):
    """
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
    # --- ffmpeg 확인 ---
    if which("ffmpeg") is None:
        print("Warning: ffmpeg not found. Audio conversion may fail.")

    print(f"🎧 Transcribing: {input_file}")
    segments, _ = model.transcribe(
        input_file,
        beam_size=beam_size,
        language="ko",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400)
    )

    # --- segment -> 문장 변환 + 오버랩 중복 제거 ---
    all_sentences = []
    last_end_time = 0
    last_sentence = None  # 이전 segment 마지막 문장
    for seg in tqdm(segments, desc="Processing segments", ncols=100):
        seg_start = max(seg.start - overlap_sec, last_end_time)
        seg_end = seg.end
        text = seg.text.strip()
        if text and len(text) > 3 and any(c.isalnum() for c in text):
            split_sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
            for s in split_sentences:
                # 이전 segment 마지막 문장과 동일하면 skip
                if s != last_sentence:
                    all_sentences.append(s)
                last_sentence = s
        last_end_time = seg_end

    # --- 중복 제거 + 너무 짧은 문장 제거 ---
    all_sentences = remove_duplicates_and_short(all_sentences, min_len=3)

    # --- 맞춤법 교정 후 중복 제거 ---
    spell_checker = spell_init()
    refined_sentences = []
    prev_sentence = None
    for i in tqdm(range(0, len(all_sentences), batch_sentence_count), desc="Spellchecking batches", ncols=100):
        batch_text = " ".join(all_sentences[i:i+batch_sentence_count])
        batch_refined = correct_text_batch(spell_checker, batch_text)
        # batch 내에서 마지막 문장과 이전 문장 비교하며 중복 제거
        for s in batch_refined:
            if s != prev_sentence:
                refined_sentences.append(s)
            prev_sentence = s

    # --- 최종 TXT 파일 저장 ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in refined_sentences:
            f.write(sentence + "\n")

    print(f"✅ Saved refined text to: {output_file}")
    return len(refined_sentences)
