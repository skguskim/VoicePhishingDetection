import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class AXEncoderWrapper(nn.Module):
    def __init__(self, model_id, freeze_layers=True, use_flash_attention=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id)
        
        model_kwargs = {}
        if use_flash_attention:
            try:
                # Check if flash_attn is installed
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = torch.float16
                print(f"⚡ Flash Attention 2 Enabled for {model_id}")
            except ImportError:
                print(f"⚠️ Flash Attention 2 requested but 'flash_attn' library not found. Falling back to default attention.")
                # Fallback to default
                pass
            except Exception as e:
                 print(f"⚠️ Flash Attention 2 initialization failed: {e}. Falling back to default.")
                 pass
            
        self.model = AutoModel.from_pretrained(model_id, config=self.config, **model_kwargs)
        
        # Output dimension: usually 768 for base models
        self.hidden_dim = self.config.hidden_size
        
        if freeze_layers:
            print("❄️ Freezing A.X. Encoder layers...")
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        # input_ids: [Batch * Seq_Len, Token_Len]
        # output: [Batch * Seq_Len, Hidden_Dim]
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token embedding (usually index 0)
        # last_hidden_state: [Batch, Token_Len, Hidden]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding

