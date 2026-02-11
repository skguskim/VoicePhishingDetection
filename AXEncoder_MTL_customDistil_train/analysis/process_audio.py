import os
import argparse
import numpy as np
from pydub import AudioSegment, silence
import noisereduce as nr
# from scipy.io import wavfile # Not strictly used with pydub conversion approach
import tempfile
import shutil
from tqdm import tqdm
import torch
import json

def load_audio(file_path):
    """오디오 파일을 로드하고 AudioSegment 객체로 반환합니다."""
    # wav 파일이 아닌 경우 pydub가 ffmpeg를 이용해 변환합니다.
    return AudioSegment.from_file(file_path)

def reduce_noise_spectral_gating(audio_segment, strength=0.8, device="cpu"):
    """
    통계적 스펙트럼 게이팅(Spectral Gating)을 사용하여 노이즈를 제거합니다.
    pydub AudioSegment를 numpy array로 변환하여 처리 후 다시 AudioSegment로 변환합니다.
    """
    # pydub AudioSegment -> numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    
    # noisereduce는 float32 타입을 선호합니다.
    # 16bit PCM 기준
    samples_float = samples.astype(np.float32)
    
    # 노이즈 감소 수행
    # use_torch=True로 설정하면 torch가 설치되어 있을 경우 GPU 사용 가능
    # device 파라미터로 명시적 지정
    try:
        reduced_noise = nr.reduce_noise(
            y=samples_float, 
            sr=audio_segment.frame_rate,
            prop_decrease=strength,
            stationary=True,
            device=device,
            use_torch=True if device != "cpu" else False
        )
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"  [Warning] Noise reduction on {device} failed, falling back to CPU: {e}")
        reduced_noise = nr.reduce_noise(
            y=samples_float, 
            sr=audio_segment.frame_rate,
            prop_decrease=strength,
            stationary=True,
            use_torch=False
        )
    
    # 다시 int16으로 변환
    reduced_samples = reduced_noise.astype(np.int16)
    
    # numpy array -> AudioSegment
    new_audio = audio_segment._spawn(reduced_samples.tobytes())
    return new_audio

def split_audio_on_silence(audio_segment, min_silence_len=700, silence_thresh=-40, keep_silence=200):
    """
    고정된 임계값을 사용하여 침묵 구간을 기준으로 오디오를 분할합니다.
    """
    # 침묵 구간 분리
    chunks = silence.split_on_silence(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence 
    )
    
    return chunks

def process_file(input_path, output_folder, strength, min_silence, fixed_thresh, thresh_offset, device):
    try:
        filename = os.path.basename(input_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # 1. 오디오 로드
        audio = load_audio(input_path)
        
        # 임계값 결정
        if fixed_thresh is not None:
            silence_thresh = fixed_thresh
        else:
            # 동적 계산 (파일별 평균 dBFS 기준)
            silence_thresh = audio.dBFS + thresh_offset
            
        # 2. 노이즈 제거
        if len(audio) > 100:
            audio = reduce_noise_spectral_gating(audio, strength=strength, device=device)
        
        # 3. 침묵 구간 분리
        chunks = split_audio_on_silence(
            audio, 
            min_silence_len=min_silence, 
            silence_thresh=silence_thresh
        )
        
        # 4. 저장
        os.makedirs(output_folder, exist_ok=True)
        
        if not chunks:
            output_path = os.path.join(output_folder, f"{filename_no_ext}_processed.wav")
            audio.export(output_path, format="wav")
        else:
            for i, chunk in enumerate(chunks):
                if len(chunk) < 500: 
                    continue
                    
                output_filename = f"{filename_no_ext}_chunk_{i:04d}.wav"
                output_path = os.path.join(output_folder, output_filename)
                chunk.export(output_path, format="wav")
                
    except Exception as e:
        print(f"  [Error] Failed to process {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Audio Noise Reduction and Segmentation Tool")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing audio files to process")
    parser.add_argument("--strength", type=float, default=0.8, help="Noise reduction strength (0.0 - 1.0)")
    parser.add_argument("--min_silence", type=int, default=700, help="Minimum silence length in ms to split")
    
    # 기존 옵션 (Offset)
    parser.add_argument("--thresh_offset", type=int, default=-16, help="dB threshold offset from average volume (Used if params_file is not provided)")
    
    # 새로운 옵션 (JSON Params)
    parser.add_argument("--params_file", type=str, default=None, help="Path to JSON file containing analysis stats (recommended)")
    
    parser.add_argument("--gpu", action="store_true", help="Use GPU for noise reduction if available")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__)) # .../AXEncoder_MTL_customDistil_train/analysis
    repo_root = os.path.dirname(script_dir) # .../AXEncoder_MTL_customDistil_train
    workspace_root = os.path.dirname(repo_root) # .../ (e.g. /home/j2hoon10)
    
    # config.py의 DATA_ROOT와 일치시키기 위해 상위 폴더의 data/preprocessing으로 설정
    base_output_dir = os.path.join(workspace_root, "data", "preprocessing")
    
    # 파라미터 로드 로직
    # 기본값 설정
    target_thresh_offset = args.thresh_offset
    fixed_thresh = None
    
    if args.params_file:
        if os.path.exists(args.params_file):
            print(f"[Info] Loading parameters from {args.params_file}")
            with open(args.params_file, 'r') as f:
                params = json.load(f)
                if "recommended_silence_thresh" in params:
                    fixed_thresh = params["recommended_silence_thresh"]
                    print(f"[Info] Using fixed silence threshold from analysis: {fixed_thresh:.2f} dB")
                else:
                    print("[Warning] 'recommended_silence_thresh' not found in JSON. Falling back to offset.")
        else:
            print(f"[Warning] Params file {args.params_file} not found. Falling back to offset.")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # GPU 설정
    device = "cpu"
    if args.gpu:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[Info] GPU detected. Using CUDA for noise reduction.")
        elif torch.backends.mps.is_available(): # Mac M1/M2 support
            device = "mps"
            print(f"[Info] MPS detected. Using MPS for noise reduction.")
        else:
            print("[Warning] GPU flag set but no GPU available. Using CPU.")
    else:
        # print("[Info] Using CPU. Add --gpu flag to use GPU if available.")
        pass

    
    input_folder_name = os.path.basename(input_dir.rstrip(os.sep))
    target_output_dir = os.path.join(base_output_dir, input_folder_name)
    
    if not os.path.exists(target_output_dir):
        os.makedirs(target_output_dir)
        print(f"Created output directory: {target_output_dir}")
        
    supported_exts = ('.wav', '.mp3', '.m4a', '.flac')
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)]
    
    if not files:
        print("No audio files found in the specified directory.")
        return
        
    print(f"Found {len(files)} audio files. Starting processing...")
    
    for f in tqdm(files, desc="Processing Audio Files", unit="file"):
        input_path = os.path.join(input_dir, f)
        
        process_file(
            input_path, 
            target_output_dir, 
            strength=args.strength, 
            min_silence=args.min_silence, 
            fixed_thresh=fixed_thresh, 
            thresh_offset=target_thresh_offset,
            device=device
        )
        
    print("Processing complete.")

if __name__ == "__main__":
    main()

