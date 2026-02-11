import os
import argparse
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import json
import torch

def load_audio(file_path):
    return AudioSegment.from_file(file_path)

def analyze_stats(input_dir, output_json, sample_rate=16000, thresh_offset=-16):
    supported_exts = ('.wav', '.mp3', '.m4a', '.flac')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)]
    
    if not files:
        print("No audio files found.")
        return

    print(f"Analyzing {len(files)} files...")
    
    db_values = []
    
    for f in tqdm(files, desc="Analyzing Audio Stats"):
        try:
            path = os.path.join(input_dir, f)
            audio = load_audio(path)
            
            # dBFS 계산
            if np.isneginf(audio.dBFS):
                continue # 침묵 파일 제외
                
            db_values.append(audio.dBFS)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
            
    if not db_values:
        print("No valid audio data found.")
        return

    # 통계 산출
    global_mean_db = np.mean(db_values)
    global_std_db = np.std(db_values)
    global_min_db = np.min(db_values)
    global_max_db = np.max(db_values)
    
    # 추천 임계값 산출 (이전에는 파일별 평균 + offset 이었음 -> 이제는 글로벌 평균 + offset)
    recommended_thresh = global_mean_db + thresh_offset
    
    stats = {
        "global_mean_db": float(global_mean_db),
        "global_std_db": float(global_std_db),
        "global_min_db": float(global_min_db),
        "global_max_db": float(global_max_db),
        "recommended_silence_thresh": float(recommended_thresh),
        "thresh_offset_used": thresh_offset,
        "files_analyzed": len(db_values)
    }
    
    print("\n[Analysis Results]")
    print(json.dumps(stats, indent=4))
    
    # 저장
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\nStats saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory of audio files to analyze")
    parser.add_argument("--output_json", default="audio_stats.json", help="Path to save the JSON output")
    parser.add_argument("--thresh_offset", type=int, default=-16, help="Offset from mean dB to determine silence threshold")
    
    args = parser.parse_args()
    
    # 입력 디렉토리 검증
    if not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        exit(1)
        
    analyze_stats(args.input_dir, args.output_json, thresh_offset=args.thresh_offset)
