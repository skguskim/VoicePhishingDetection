import sys
import os
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("🔍 Verifying Imports...")
try:
    from AXEncoder_MTL_customDistil_train import config
    from AXEncoder_MTL_customDistil_train.dataset import PhishingDataset, collate_fn
    from AXEncoder_MTL_customDistil_train.model.ax_encoder import AXEncoderWrapper
    from AXEncoder_MTL_customDistil_train.model.hierarchical_model import HierarchicalPhishingModel
    print("✅ Imports Successful!")
except Exception as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

print("\n🏗️ Verifying Model Initialization (CPU)...")
try:
    model = HierarchicalPhishingModel("skt/A.X-Encoder-base", hidden_dim=64, num_layers=1)
    print("✅ Model Initialized Successfully!")
except Exception as e:
    print(f"❌ Model Initialization Failed: {e}")
    sys.exit(1)

print("\n🎉 Codebase Verification Passed!")
