import os
import json
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

try:
    from AXEncoder_MTL_customDistil_train import config
except ImportError:
    import config

def create_dataset():
    print("🚀 Creating Dataset from Transcripts...")
    
    data = []
    
    # Iterate through all categories defined in config
    for category, label in config.CATEGORY_TO_LABEL.items():
        category_dir = os.path.join(config.DATA_ROOT, category)
        
        if not os.path.exists(category_dir):
            continue
            
        print(f"  Processing Category: {category} (Label: {label})")
        
        # New approach: Find all transcription JSONs directly in category folder
        json_files = glob.glob(os.path.join(category_dir, "*_transcription.json"))
        
        if not json_files:
            # print(f"    ⚠️ Warning: No transcriptions found in {category}")
            continue
            
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    transcripts = json.load(f)
                
                # Derive conversation ID from filename
                # Filename format: {conv_id}_transcription.json
                filename = os.path.basename(json_path)
                conv_id = filename.replace("_transcription.json", "")
                
                data.append({
                    "category": category,
                    "conversation_id": conv_id,
                    "json_path": json_path,
                    "label": label,
                    "num_chunks": len(transcripts)
                })
            except Exception as e:
                print(f"    ❌ Error reading {json_path}: {e}")
            
    if not data:
        print("❌ No data found!")
        return

    df = pd.DataFrame(data)
    print(f"\n📊 Total Samples: {len(df)}")
    if not df.empty:
        print(df['label'].value_counts())
    
        # Split Train/Val
        # Stratify might fail if some labels have very few samples (e.g. 1)
        try:
            train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
        except ValueError:
            print("  ⚠️ Warning: Not enough samples for stratification. Switching to random split.")
            train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        
        # Save CSVs
        output_dir = os.path.join(PROJECT_ROOT, "AXEncoder_MTL_customDistil_train", "data")
        os.makedirs(output_dir, exist_ok=True)
        
        train_csv = os.path.join(output_dir, "train.csv")
        val_csv = os.path.join(output_dir, "val.csv")
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        
        print(f"\n✅ Saved Dataset to {output_dir}")
        print(f"  - Train: {len(train_df)}")
        print(f"  - Val:   {len(val_df)}")

if __name__ == "__main__":
    create_dataset()
