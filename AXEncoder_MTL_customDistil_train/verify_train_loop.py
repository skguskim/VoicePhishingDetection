
import os
import sys
import torch
from unittest.mock import MagicMock


# Add project root to path (one level up for direct package import)
PROJECT_ROOT = os.path.dirname(os.getcwd())
sys.path.append(PROJECT_ROOT)
# Also add current dir
sys.path.append(os.getcwd())

# Mock WandB to avoid login prompt/error during automated test
import wandb
wandb.init = MagicMock()
wandb.log = MagicMock()

try:
    try:
        from AXEncoder_MTL_customDistil_train import config
        from AXEncoder_MTL_customDistil_train.train import train_model
        from AXEncoder_MTL_customDistil_train.dataset import PhishingDataset, collate_fn
    except ImportError:
        # Local import fallback
        import config
        from train import train_model
        from dataset import PhishingDataset, collate_fn
    
    # Overwrite config for dry run
    config.EPOCHS = 1
    config.BATCH_SIZE = 2
    config.USE_FLASH_ATTENTION = False # Fallback test
    
    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Modify Dataset to return dummy data since real data is missing
def dummy_getitem(self, idx):
    # Return random tensors mimicking structure
    # Input: [Seq_Len, Token_Len]
    # We simulate 2 chunks of length 128
    encoded_chunks = []
    for _ in range(2):
        encoded_chunks.append({
            'input_ids': torch.randint(0, 1000, (128,)),
            'attention_mask': torch.ones(128)
        })
    return {
        'chunks': encoded_chunks,
        'label': torch.tensor(1.0 if idx % 2 == 0 else 0.0) # Alternate labels
    }

# Patch Dataset
PhishingDataset.__getitem__ = dummy_getitem
PhishingDataset.__len__ = lambda self: 10 # 10 samples for dry run

# Create dummy CSVs to pass file existence checks
os.makedirs("data", exist_ok=True)
with open("data/train.csv", "w") as f:
    f.write("dummy_header\n")
with open("data/val.csv", "w") as f:
    f.write("dummy_header\n")

print("🚀 Starting Dry-Run Training...")
try:
    train_model()
    print("✅ Dry-run training completed successfully")
    
    # Verify checkpoint creation
    if os.path.exists("AXEncoder_MTL_customDistil_train/model/last_checkpoint.pt"):
        print("✅ Checkpoint created")
    else:
        print("❌ Checkpoint NOT created")
        
except Exception as e:
    print(f"❌ Dry-run failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup dummy files
    pass
