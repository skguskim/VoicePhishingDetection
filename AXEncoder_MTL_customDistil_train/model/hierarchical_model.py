import torch
import torch.nn as nn
from .ax_encoder import AXEncoderWrapper

class HierarchicalPhishingModel(nn.Module):
    def __init__(self, text_encoder_id, hidden_dim=256, num_layers=2, use_flash_attention=False):
        super().__init__()
        
        # 1. Chunk Encoder (Pretrained Text Model)
        self.chunk_encoder = AXEncoderWrapper(
            text_encoder_id, 
            freeze_layers=True, 
            use_flash_attention=use_flash_attention
        )
        input_dim = self.chunk_encoder.hidden_dim
        
        # 2. Context Aggregator (Bi-LSTM)
        # Input: Sequence of Chunk Embeddings [Batch, Seq_Len, Input_Dim]
        self.aggregator = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Classifier Head
        # Bi-LSTM output is 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1) # Probability logit
        )
        
    def forward(self, input_ids, attention_mask, sequence_mask):
        # input_ids: [Batch, Max_Seq, Token_Len]
        # sequence_mask: [Batch, Max_Seq] (Which chunks are valid)
        
        batch_size, max_seq, token_len = input_ids.size()
        
        # --- Chunk Unpadding ---
        # Flatten input to [Batch * Max_Seq, Token_Len]
        input_ids_flat = input_ids.view(-1, token_len)
        attention_mask_flat = attention_mask.view(-1, token_len)
        
        # Create mask for valid chunks (from sequence_mask)
        # sequence_mask is 1 for valid chunks, 0 for padded chunks
        valid_chunk_mask = sequence_mask.view(-1).bool()
        
        # Filter for only valid chunks
        valid_input_ids = input_ids_flat[valid_chunk_mask]
        valid_attention_mask = attention_mask_flat[valid_chunk_mask]
        
        # Check if we have any valid chunks (edge case)
        if valid_input_ids.size(0) == 0:
             # Should practically not happen with correct data, but safe default
             chunk_embeddings = torch.zeros(batch_size, max_seq, self.chunk_encoder.hidden_dim, device=input_ids.device)
             if hasattr(self.chunk_encoder, "config") and hasattr(self.chunk_encoder.config, "torch_dtype"):
                 if self.chunk_encoder.config.torch_dtype == torch.float16:
                     chunk_embeddings = chunk_embeddings.half()
        else:
            # Pass ONLY valid chunks through Encoder
            # [Total_Valid_Chunks, Enc_Dim]
            valid_chunk_embeddings = self.chunk_encoder(valid_input_ids, valid_attention_mask)
            
            # Prepare full embedding tensor (initialized with zeros)
            # We need to scatter the valid embeddings back to their original positions
            # [Batch * Max_Seq, Enc_Dim]
            scatter_dim = valid_chunk_embeddings.size(-1)
            full_embeddings_flat = torch.zeros(
                batch_size * max_seq, 
                scatter_dim, 
                device=input_ids.device, 
                dtype=valid_chunk_embeddings.dtype
            )
            
            # Create indices for scatter
            # We need indices where valid_chunk_mask is True. 
            # These are [0, 1, 2, 5, 6...] indices in the flattened batch dimension.
            valid_indices = torch.nonzero(valid_chunk_mask).view(-1)
            
            # Scatter updates
            # full_embeddings_flat[valid_indices] = valid_chunk_embeddings
            full_embeddings_flat.index_copy_(0, valid_indices, valid_chunk_embeddings)
            
            # Reshape back to sequence
            # [Batch, Max_Seq, Enc_Dim]
            chunk_embeddings = full_embeddings_flat.view(batch_size, max_seq, -1)
        
        # Pass through LSTM
        # output: [Batch, Max_Seq, 2*Hidden]
        # LSTM might require float32 if not using specific kernel, or ensure input/weights match
        # If chunk_embeddings is half/bfloat16, we might need to cast or ensure LSTM is also half
        # However, nn.LSTM usually expects float32 unless specifically cast.
        # Safer to cast to float32 for LSTM stability in AMP if issues arise, 
        # or rely on autocast to handle it.
        # The error "ValueError: input must have the type torch.float32, got type torch.bfloat16" 
        # suggests autocast isn't covering this or the weights are float32.
        
        lstm_out, _ = self.aggregator(chunk_embeddings.float())
        
        # Extract last valid state via sequence_mask or just Max Pooling
        # Max Pooling over time is robust for "detecting if phishing happened anywhere"
        
        # Mask out invalid steps (padded chunks)
        # sequence_mask: [Batch, Max_Seq] -> [Batch, Max_Seq, 1]
        mask_expanded = sequence_mask.unsqueeze(-1)  # bool tensor
        
        # For samples with ALL chunks invalid (empty transcripts), we need special handling
        # Check which samples have at least one valid chunk
        has_valid_chunk = sequence_mask.any(dim=1)  # [Batch]
        
        # Apply mask: use a very negative value for invalid positions
        # Use masked_fill which is safer than torch.where with -inf
        masked_out = lstm_out.clone()
        masked_out = masked_out.masked_fill(~mask_expanded, -1e4)
        
        # Max Pooling over sequence dimension
        # [Batch, 2*Hidden]
        context_vector, _ = torch.max(masked_out, dim=1)
        
        # For samples with NO valid chunks, replace with zeros to avoid -inf propagation
        context_vector = torch.where(
            has_valid_chunk.unsqueeze(-1),
            context_vector,
            torch.zeros_like(context_vector)
        )
        
        # Classifier
        # [Batch, 1]
        logits = self.classifier(context_vector)
        
        return logits
