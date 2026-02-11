import os
import json
import glob
from tqdm import tqdm
from faster_whisper import WhisperModel
import torch
import sys
import re

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from AXEncoder_MTL_customDistil_train import config
except ImportError:
    # If running directly from within the folder, try relative import
    import config

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def transcribe_dataset(input_root, model_id):
    print(f"🚀 Initializing Faster-Whisper Model: {model_id}")
    device = get_device()
    compute_type = "bfloat16" if device == "cuda" else "int8"
    
    try:
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
        print(f"✅ Model loaded on {device} with {compute_type} precision.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Find all categorization folders
    categories = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for category in categories:
        category_dir = os.path.join(input_root, category)
        print(f"\n📂 Processing Category: {category}")
        
        # Get all chunk files
        all_files = sorted(glob.glob(os.path.join(category_dir, "*_chunk_*.wav")))
        if not all_files:
            print(f"  ⚠️ No chunk files found in {category}")
            continue

        # Group by conversation (filename prefix)
        # Pattern: {conversation_id}_chunk_{index}.wav
        # We need to extract {conversation_id}
        # Be careful if conversation_id contains "_chunk_" (unlikely but possible)
        # or if conversation_id has underscores.
        # Regex strategy: (.*)_chunk_\d+\.wav
        
        conversation_groups = {}
        for file_path in all_files:
            filename = os.path.basename(file_path)
            match = re.match(r"(.*)_chunk_\d+\.wav", filename)
            if match:
                conv_id = match.group(1)
                if conv_id not in conversation_groups:
                    conversation_groups[conv_id] = []
                conversation_groups[conv_id].append(file_path)
        
        print(f"  Found {len(conversation_groups)} unique conversations.")

        # Process each conversation group
        for conv_id, chunk_paths in tqdm(conversation_groups.items(), desc=f"Transcribing {category}"):
            # Output JSON path: data/preprocessing/{category}/{conv_id}_transcription.json
            output_json_path = os.path.join(category_dir, f"{conv_id}_transcription.json")
            
            if os.path.exists(output_json_path):
                continue
            
            transcriptions = []
            
            # Sort chunks by name (index) to ensure order
            chunk_paths.sort()
            
            for chunk_file in chunk_paths:
                chunk_name = os.path.basename(chunk_file)
                try:
                    segments, info = model.transcribe(chunk_file, beam_size=5, language="ko")
                    text = " ".join([segment.text for segment in segments]).strip()
                    
                    transcriptions.append({
                        "file": chunk_name,
                        "text": text,
                        "duration": info.duration
                    })
                except Exception as e:
                    print(f"  ⚠️ Error transcribing {chunk_name}: {e}")
                    transcriptions.append({
                        "file": chunk_name,
                        "text": "",
                        "error": str(e)
                    })
            
            # Save to JSON
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    print("\n🎉 All Transcriptions Completed!")

if __name__ == "__main__":
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ Data root not found: {config.DATA_ROOT}")
        print("Please run the audio processing pipeline first.")
    else:
        transcribe_dataset(config.DATA_ROOT, config.STT_MODEL_ID)
