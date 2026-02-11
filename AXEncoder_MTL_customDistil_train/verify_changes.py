
import sys
import os
import torch
from transformers import logging

logging.set_verbosity_error() # Suppress transformer warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

print(f"Path: {sys.path}")

try:
    try:
        from AXEncoder_MTL_customDistil_train import config
        from AXEncoder_MTL_customDistil_train.model.hierarchical_model import HierarchicalPhishingModel
        print("✅ Imports successful (package style)")
    except ImportError:
        import config
        from model.hierarchical_model import HierarchicalPhishingModel
        print("✅ Imports successful (local style)")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("🔍 Checking Model instantiation with Flash Attention request...")
try:
    # Request FA2 even if not installed to test fallback
    # We must ensure config.TEXT_ENCODER_ID exists
    model_id = config.TEXT_ENCODER_ID
    model = HierarchicalPhishingModel(model_id, use_flash_attention=True)
    print("✅ Model instantiated successfully (Check output for fallback warning)")
except Exception as e:
    print(f"❌ Model instantiation failed: {e}")
    # Print full traceback for debugging
    import traceback
    traceback.print_exc()
    sys.exit(1)
