import torch
import os
import sys
import json
from faster_whisper import WhisperModel

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from AXencoder import config
    from AXencoder.hierarchical_model import HierarchicalPhishingModel
except ImportError:
    import config
    from hierarchical_model import HierarchicalPhishingModel

from transformers import AutoTokenizer

class PhishingDetector:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing Detector on {self.device}...")
        
        # 1. Load STT Model
        print(f"  - Loading STT Model: {config.STT_MODEL_ID}...")
        compute_type = "bfloat16" if self.device == "cuda" else "int8"
        self.stt_model = WhisperModel(config.STT_MODEL_ID, device=self.device, compute_type=compute_type)
        
        # 2. Load Phishing Model
        print(f"  - Loading Phishing Classifier: {config.TEXT_ENCODER_ID} (Base)...")
        self.model = HierarchicalPhishingModel(config.TEXT_ENCODER_ID).to(self.device)
        
        if model_path and os.path.exists(model_path):
            print(f"  - Loading Weights from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            # Support both full checkpoint (from train.py) and raw state_dict formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("  ⚠️ Warning: No trained model weights found. Using random initialization (for testing).")
            
        self.model.eval()
        
        # 3. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER_ID)
        
    def predict(self, audio_path):
        # Step 1: STT
        print(f"\n🎧 Processing Audio: {audio_path}")
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=5, language="ko")
        text_chunks = [segment.text for segment in segments]
        print(text_chunks)
        full_text = " ".join(text_chunks)
        print(f"  📝 Transcribed: {full_text[:50]}...")
        
        if not text_chunks:
            return 0.0, "Normal (No Speech)"
            
        # Step 2: Preprocess for Model
        # Truncate to max seq len
        if len(text_chunks) > config.MAX_SEQ_LEN:
            text_chunks = text_chunks[-config.MAX_SEQ_LEN:]
            
        # Tokenize chunks
        input_ids_list = []
        attention_mask_list = []
        
        for chunk in text_chunks:
            encoded = self.tokenizer(
                chunk,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids_list.append(encoded['input_ids'])
            attention_mask_list.append(encoded['attention_mask'])
            
        # Stack [1, Seq, Token]
        input_ids = torch.stack(input_ids_list).to(self.device).unsqueeze(0) # Add batch dim
        attention_mask = torch.stack(attention_mask_list).to(self.device).unsqueeze(0)
        
        # Sequence mask [1, Seq] (All valid here)
        sequence_mask = torch.ones(1, len(text_chunks), dtype=torch.bool).to(self.device)
        
        # Step 3: Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, sequence_mask)
            prob = torch.sigmoid(logits).item()
            
        label = "Phishing 🚨" if prob > 0.5 else "Normal ✅"
        return prob, label

if __name__ == "__main__":
    # Example Usage
    detector = PhishingDetector()
    
    # Test with a dummy file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        prob, label = detector.predict(audio_file)
        print(f"\n🔍 Result: {label} ({prob*100:.2f}%)")
