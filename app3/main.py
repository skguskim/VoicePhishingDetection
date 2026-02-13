from fastapi import FastAPI, UploadFile, File, Form
import os, wave, sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from faster_whisper import WhisperModel
import torchaudio
import numpy as np

from AXencoder import config
from AXencoder.hierarchical_model import HierarchicalPhishingModel

app = FastAPI()

APP_DIR = Path(__file__).resolve().parent       
STATIC_DIR = APP_DIR / "static"                  
BASE_DIR = APP_DIR / "data"                       
BASE = str(BASE_DIR)                              
os.makedirs(BASE, exist_ok=True)


# =========================
# (A) Student 모델 import 경로 추가 + import
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "bfloat16" if DEVICE == "cuda" else "int8"   # CPU면 int8 권장
WHISPER = WhisperModel(config.STT_MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[Whisper] ready")
# =========================
# (B) Student 모델 생성에 필요한 값 추출
# =========================

model = HierarchicalPhishingModel(config.TEXT_ENCODER_ID).to(DEVICE)

# =========================
# (C) Student 가중치 로드 (state_dict)
# =========================
checkpoint = torch.load("AXencoder/best_model.pt", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER_ID)
print("[Model] ready")

# =========================
# (D) Whisper STT 로드 
# =========================
print("STATIC DIR:", STATIC_DIR)

def session_dirs(sid: str):
    b = f"{BASE}/session_{sid}"
    d5 = f"{b}/wav_5s"
    d15 = f"{b}/wav_15s"
    os.makedirs(d5, exist_ok=True)
    os.makedirs(d15, exist_ok=True)
    return d5, d15

def concat_wav(in_paths, out_path):
    with wave.open(in_paths[0], "rb") as w:
        params = w.getparams()
        frames = [w.readframes(w.getnframes())]
    for p in in_paths[1:]:
        with wave.open(p, "rb") as w:
            frames.append(w.readframes(w.getnframes()))
    with wave.open(out_path, "wb") as w:
        w.setparams(params)
        for f in frames:
            w.writeframes(f)
            
def preprocess_wav_to_16k_mono(in_path: str, out_path: str):
    wav, sr = torchaudio.load(in_path)  # [ch, time]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono

    # resample to 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # band-pass 비슷하게: highpass + lowpass (환경에 따라 컷오프 튜닝)
    wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff_freq=80.0)
    wav = torchaudio.functional.lowpass_biquad(wav, sr, cutoff_freq=7800.0)

    # 클리핑 방지용 정규화(선택)
    peak = wav.abs().max().clamp(min=1e-6)
    wav = wav / peak

    torchaudio.save(out_path, wav, sr)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            f"<h3>index.html not found</h3><p>Expected: {index_path}</p>",
            status_code=404
        )
    return index_path.read_text(encoding="utf-8")

@app.post("/upload_wav")
async def upload_wav(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    index: str = Form(...)
):
    index = int(index)

    d5, d15 = session_dirs(session_id)

    data = await file.read()
    p = f"{d5}/{index:03d}.wav"
    with open(p, "wb") as f:
        f.write(data)

    made_15s = False
    out_15s_path = None

    # 5초 3개 -> 15초 1개
    if index >= 2:
        paths = [f"{d5}/{i:03d}.wav" for i in range(index - 2, index + 1)]
        if all(os.path.exists(x) for x in paths):
            out = f"{d15}/{index-2:03d}_15s.wav"
            out_15s_path = out
            if not os.path.exists(out):
                concat_wav(paths, out)
            made_15s = True

    # =========================
    # (E) STT + 모델 추론
    # =========================

    analyze_path = out_15s_path if made_15s and out_15s_path else p

    stt_text = ""
    pred = None
    score = None

    #try:
    # (1) 전처리된 파일 경로 만들기
    denoised_path = f"{d5}/{Path(analyze_path).stem}_denoise16k.wav"
    preprocess_wav_to_16k_mono(analyze_path, denoised_path)


    # (2) faster-whisper transcribe (segments)
    segments, _ = WHISPER.transcribe(
        denoised_path,
        beam_size=5,                     
        language="ko",
        condition_on_previous_text=False,
        vad_filter=True,                  # VAD로 무음/잡음 구간 필터
        temperature=0.0,
        no_speech_threshold=0.6
        # repetition_penalty 는 faster-whisper 기본 시그니처엔 없는 경우가 많습니다.
        # (버전/백엔드에 따라 지원이 다름) -> 아래 "반복 억제" 대안 참고
    )
    text_chunks = [segment.text for segment in segments]
    print(text_chunks, len(text_chunks))
    stt_text = " ".join(text_chunks)

    if not text_chunks:
        return 0.0, "Normal (No Speech)"
    
    if len(text_chunks) > config.MAX_SEQ_LEN:
        text_chunks = text_chunks[-config.MAX_SEQ_LEN:]
        
    # Tokenize chunks
    input_ids_list = []
    attention_mask_list = []
    
    for chunk in text_chunks:
        encoded = tokenizer(
            chunk,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])
        
    # Stack [1, Seq, Token]
    input_ids = torch.stack(input_ids_list).to(DEVICE).permute(1, 0, 2)
    print(input_ids.shape)

    attention_mask = torch.stack(attention_mask_list).to(DEVICE).permute(1, 0, 2)
    print(attention_mask.shape)

    # Sequence mask [1, Seq] (All valid here)
    sequence_mask = torch.ones(1, len(text_chunks), dtype=torch.bool).to(DEVICE)
    print(sequence_mask.shape)

    # Step 3: Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask, sequence_mask)
        prob = torch.sigmoid(logits).item()
        
    label = "Phishing 🚨" if prob > 0.5 else "Normal ✅"
        
    
    '''except Exception as e:
        return {
            "saved": index,
            "session_id": session_id,
            "analyzed_path": analyze_path,
            "stt_text": stt_text,
            "pred": pred,
            "score": score,
            "error": str(e),
        }
        '''
    return {
        "saved": index,
        "session_id": session_id,
        "analyzed_path": analyze_path,
        "stt_text": stt_text,
        "pred": label,
        "score": prob,
        "made_15s": made_15s,
    }