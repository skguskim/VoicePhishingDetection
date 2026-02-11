import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from transformers import AutoTokenizer
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from AXEncoder_MTL_customDistil_train import config
except ImportError:
    import config

class PhishingDataset(Dataset):
    def __init__(self, csv_path, tokenizer_id=config.TEXT_ENCODER_ID, max_seq_len=config.MAX_SEQ_LEN):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw_path = row['json_path']
        label = row['label']
        
        # Handle Windows paths from CSV
        # Extract filename and category to reconstruct Linux path
        filename = os.path.basename(raw_path.replace("\\", "/"))
        category = row['category']
        
        # Construct valid path
        json_path = os.path.join(config.DATA_ROOT, category, filename)
        
        # Load transcriptions
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
        except Exception as e:
            # print(f"Error loading {json_path}: {e}") # Reduce noise
            transcripts = []
            
        # Extract text chunks
        # Depending on the model design, we might want to concatenate valid texts
        texts = [item['text'] for item in transcripts if item['text'].strip()]
        
        # Truncate sequence length (Chunks)
        if len(texts) > self.max_seq_len:
            # Strategy: Late window (most recent context) or random window?
            # For phishing, crucial info might be anywhere. 
            # Let's take the LAST max_seq_len chunks as they might contain the 'action' (transfer money etc.)
            # Or simplified: just take the last N
            texts = texts[-self.max_seq_len:]
            
        # Tokenize
        # We need to tokenize EACH chunk independently for the hierarchical model
        # Output: [Seq_Len, Token_Len]
        
        encoded_chunks = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128, # Short text chunks
                return_tensors='pt'
            )
            encoded_chunks.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
            
        return {
            'chunks': encoded_chunks,
            'label': torch.tensor(label, dtype=torch.float)
        }

def collate_fn(batch):
    # Batch is a list of dicts
    labels = torch.stack([item['label'] for item in batch])
    
    # Handle variable sequence lengths (number of chunks)
    # We need to pad the SEQUENCE dimension
    
    batch_size = len(batch)
    max_chunks_in_batch = max(len(item['chunks']) for item in batch)
    
    # Feature dim depends on tokenizer (usually 128 here)
    if max_chunks_in_batch == 0:
        # Handle empty batch edge case
        return None
        
    token_len = batch[0]['chunks'][0]['input_ids'].size(0) if batch[0]['chunks'] else 128
    
    # Initialize tensors [Batch, Max_Seq, Token_Len]
    padded_input_ids = torch.zeros(batch_size, max_chunks_in_batch, token_len, dtype=torch.long)
    padded_attention_mask = torch.zeros(batch_size, max_chunks_in_batch, token_len, dtype=torch.long)
    
    # Mask for the sequence logic (which chunks are real)
    # [Batch, Max_Seq]
    sequence_mask = torch.zeros(batch_size, max_chunks_in_batch, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        chunks = item['chunks']
        num_chunks = len(chunks)
        
        for j, chunk in enumerate(chunks):
            padded_input_ids[i, j] = chunk['input_ids']
            padded_attention_mask[i, j] = chunk['attention_mask']
            sequence_mask[i, j] = True
            
    return {
        'input_ids': padded_input_ids,          # [B, S, T]
        'attention_mask': padded_attention_mask, # [B, S, T] (Word-level mask)
        'sequence_mask': sequence_mask,          # [B, S]    (Chunk-level mask)
        'labels': labels                         # [B]
    }
