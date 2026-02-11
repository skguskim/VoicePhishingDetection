import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from AXEncoder_MTL_customDistil_train import config
    from AXEncoder_MTL_customDistil_train.dataset import PhishingDataset, collate_fn
    from AXEncoder_MTL_customDistil_train.model.hierarchical_model import HierarchicalPhishingModel
    from AXEncoder_MTL_customDistil_train.utils import set_seed
except ImportError:
    # Relative import fallback
    import config
    from dataset import PhishingDataset, collate_fn
    from model.hierarchical_model import HierarchicalPhishingModel
    from utils import set_seed

def train_model():
    set_seed(42)
    print("🚀 Initializing Training Pipeline...")
    
    # Check for processed data
    train_csv = os.path.join(PROJECT_ROOT, "AXEncoder_MTL_customDistil_train", "data", "train.csv")
    val_csv = os.path.join(PROJECT_ROOT, "AXEncoder_MTL_customDistil_train", "data", "val.csv")
    
    if not os.path.exists(train_csv):
        print(f"❌ Training data not found at {train_csv}")
        # print("Please run 'create_dataset.py' first.") # Suppress as user knows
        # return # User wants to proceed anyway with potential missing files handled in dataset
    
    # --- WandB Init ---
    try:
        import wandb
        wandb.init(project=config.WANDB_PROJECT, name=f"run_{config.TEXT_ENCODER_ID.replace('/', '_')}")
        print("✅ WandB Initialized")
    except ImportError:
        print("⚠️ WandB not installed. Skipping logging.")
        wandb = None
    except Exception as e:
        print(f"⚠️ WandB initialization failed: {e}. Skipping logging.")
        wandb = None

    # 1. Dataset & DataLoader
    print("📚 Loading Datasets...")
    train_dataset = PhishingDataset(train_csv)
    val_dataset = PhishingDataset(val_csv)
    
    # 2. Model
    print(f"🏗️ Building Model (Base: {config.TEXT_ENCODER_ID})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HierarchicalPhishingModel(
        config.TEXT_ENCODER_ID, 
        use_flash_attention=config.USE_FLASH_ATTENTION
    ).to(device)
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() # For binary classification
    
    # Scaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)
    
    # --- Checkpointing Setup ---
    checkpoint_dir = os.path.join(PROJECT_ROOT, "AXEncoder_MTL_customDistil_train", "model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    start_epoch = 0
    best_val_f1 = 0.0
    
    # Resume from checkpoint if exists
    if os.path.exists(last_checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and config.USE_AMP:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        print(f"   Resumed at Epoch {start_epoch+1}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 4. Training Loop
    print(f"🔥 Starting Training for {config.EPOCHS} epochs on {device}...")
    if config.USE_AMP:
        print("⚡ Mixed Precision Training Enabled")
        
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        for batch in progress_bar:
            if batch is None: continue
            
            # Robust Batch Unpacking
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sequence_mask = batch['sequence_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                logits = model(input_ids, attention_mask, sequence_mask)
                logits = logits.squeeze(-1) # [Batch]
                loss = criterion(logits, labels)
            
            # Detailed NaN debugging (only for first batch)
            if torch.isnan(loss) or torch.isinf(loss) or (hasattr(progress_bar, 'n') and progress_bar.n == 0):
                # Check input data
                print(f"\n🔍 Debug Info (Batch {progress_bar.n if hasattr(progress_bar, 'n') else 0}):")
                print(f"  Input IDs: shape={input_ids.shape}, has_nan={torch.isnan(input_ids.float()).any()}")
                print(f"  Attention Mask: shape={attention_mask.shape}, has_nan={torch.isnan(attention_mask.float()).any()}")
                print(f"  Sequence Mask: shape={sequence_mask.shape}, has_nan={torch.isnan(sequence_mask.float()).any()}")
                
                # Check sequence_mask distribution
                chunks_per_sample = sequence_mask.sum(dim=1)
                print(f"  Chunks per sample: {chunks_per_sample.tolist()}")
                print(f"  Samples with 0 chunks: {(chunks_per_sample == 0).sum().item()}")
                
                print(f"  Labels: {labels}, has_nan={torch.isnan(labels).any()}")
                
                # Check intermediate outputs with no_grad to avoid affecting training
                with torch.no_grad():
                    # Step through model manually
                    batch_size, max_seq, token_len = input_ids.size()
                    input_ids_flat = input_ids.view(-1, token_len)
                    attention_mask_flat = attention_mask.view(-1, token_len)
                    valid_chunk_mask = sequence_mask.view(-1).bool()
                    
                    valid_input_ids = input_ids_flat[valid_chunk_mask]
                    valid_attention_mask = attention_mask_flat[valid_chunk_mask]
                    
                    print(f"  Valid chunks total: {valid_input_ids.size(0)}")
                    
                    # Check encoder output
                    encoder_out = model.chunk_encoder(valid_input_ids, valid_attention_mask)
                    print(f"  Encoder output: shape={encoder_out.shape}, has_nan={torch.isnan(encoder_out).any()}, range=[{encoder_out.min():.3f}, {encoder_out.max():.3f}]")
                
                print(f"  Logits: min={logits.min() if not torch.isnan(logits).all() else 'all_nan'}, max={logits.max() if not torch.isnan(logits).all() else 'all_nan'}, has_nan={torch.isnan(logits).any()}")
                print(f"  Loss: {loss.item()}")
                
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError("Training stopped due to NaN/Inf loss")
            
            # Backward with Scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            current_loss = loss.item()
            progress_bar.set_postfix({'loss': current_loss})
            
            if wandb:
                wandb.log({"train_batch_loss": current_loss})
            
        # Validation
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds)
        avg_train_loss = train_loss/len(train_loader) if len(train_loader) > 0 else 0
        
        print(f"\n📢 Epoch {epoch+1} Summary:")
        print(f"  - Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  - Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Log to WandB
        if wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1
            })
        
        # --- Checkpointing ---
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if config.USE_AMP else None,
            'best_val_f1': best_val_f1
        }
        
        # Save Last Checkpoint (Overwrite)
        torch.save(checkpoint_state, last_checkpoint_path)
        
        # Save Best Model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Update best metric in checkpoint state
            checkpoint_state['best_val_f1'] = best_val_f1 
            torch.save(checkpoint_state, best_model_path) # Save full state to allow resumption from best? 
            # Usually best_model.pt is just weights, but having full state is safer.
            # Requirement: "가장 마지막 학습된 모델 데이터와 가장 성능이 좋았던 학습된 모델 데이터만 저장"
            # So overwriting best_model.pt is correct.
            
            print(f"  💾 Saved Best Model (F1: {best_val_f1:.4f})")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=config.USE_AMP):
        for batch in tqdm(dataloader, desc="[Valid]"):
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sequence_mask = batch['sequence_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, sequence_mask)
            logits = logits.squeeze(-1)
            
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    return avg_loss, acc, f1

if __name__ == "__main__":
    train_model()
