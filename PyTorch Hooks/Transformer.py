import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Split into multiple heads: (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        
        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # (B, num_heads, T, T) @ (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        out = attn @ V
        
        # Concatenate heads: (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        
        # Feedforward with residual
        x = x + self.dropout(self.ff(self.ln2(x)))
        
        return x


class TransformerLanguageModel(nn.Module):
    """Transformer language model"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        
        # Positional embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.pos_embedding(pos)  # (T, C)
        
        # Combine embeddings
        x = tok_emb + pos_emb
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Take last token and project to vocabulary
        logits = self.head(x[:, -1, :])  # (B, vocab_size)
        
        return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx if idx.shape[1] <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Get predictions
            logits = self(idx_cond)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def save_model(self, filepath, stoi, itos, train_loss=None, val_loss=None):
        """Save model with all necessary information"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hyperparameters': {
                'vocab_size': self.token_embedding.num_embeddings,
                'embed_dim': self.embed_dim,
                'num_heads': self.blocks[0].attn.num_heads,
                'num_layers': len(self.blocks),
                'ff_dim': self.blocks[0].ff.fc1.out_features,
                'max_seq_len': self.max_seq_len,
                'dropout': self.dropout.p
            },
            'vocab': {
                'stoi': stoi,
                'itos': itos
            },
            'losses': {
                'train_loss': train_loss,
                'val_loss': val_loss
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath, device='cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with saved hyperparameters
        model = TransformerLanguageModel(**checkpoint['hyperparameters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Extract vocab and losses
        stoi = checkpoint['vocab']['stoi']
        itos = checkpoint['vocab']['itos']
        losses = checkpoint['losses']
        
        print(f"Model loaded from {filepath}")
        print(f"Train loss: {losses['train_loss']:.4f}, Val loss: {losses['val_loss']:.4f}")
        
        return model, stoi, itos


def get_batch(data, context_length, batch_size, device):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix]).to(device)
    y = data[ix + context_length].to(device)
    return x, y


def estimate_loss(model, data, context_length, batch_size, device, eval_iters=100):
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, context_length, batch_size, device)
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"DEVICE: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*60}\n")
    
    # Set CPU threads if using CPU
    if device.type == 'cpu':
        torch.set_num_threads(8)
        print(f"Using {torch.get_num_threads()} CPU threads\n")
    
    # Hyperparameters
    batch_size = 128  # Reduced for better speed
    num_epochs = 40
    steps_per_epoch = 1000
    eval_interval = 1
    learning_rate = 2e-3
    
    # Model hyperparameters
    context_length = 64        # Reduced from 256 for MUCH faster training
    embed_dim = 256
    num_heads = 4
    num_layers = 3
    ff_dim = embed_dim * 4
    dropout = 0.05
    max_seq_len = 64
    
    # Timing variables
    start_time = time.time()
    print_interval = 100
    total_tokens = 0
    total_steps = 0
    prev_val_loss = None
    
    # Load data
    text = ""
    books = [
        'Dataset/tinychat.txt'
        # Add more books here
    ]

    for book_path in books:
        with open(book_path, 'r', encoding='utf-8') as f:
            text += f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset size: {len(train_data):,} tokens")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {steps_per_epoch * num_epochs:,}")
    print(f"Context length: {context_length}")
    print(f"Batch size: {batch_size}")
    print(f"Embed dim: {embed_dim}")
    print(f"Num heads: {num_heads}")
    print(f"Num layers: {num_layers}")
    
    # Create model and move to device
    model = TransformerLanguageModel(
        vocab_size, embed_dim, num_heads, num_layers, 
        ff_dim, max_seq_len, dropout
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        for step in range(steps_per_epoch):
            xb, yb = get_batch(train_data, context_length, batch_size, device)
            
            # Forward pass
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update counters
            total_tokens += batch_size * context_length
            total_steps += 1
            epoch_loss += loss.item()
            
            # Print periodic updates
            if step % print_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
                tokens_per_sec = steps_per_sec * batch_size * context_length
                current_lr = scheduler.get_last_lr()[0]
                print(f"  step {step}/{steps_per_epoch} | "
                      f"loss: {loss.item():.4f} | "
                      f"lr: {current_lr:.6f} | "
                      f"steps/s: {steps_per_sec:.2f} | "
                      f"tokens/s: {tokens_per_sec:.0f}")
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / steps_per_epoch
        total_time = time.time() - start_time
        
        print(f"\n  Epoch {epoch + 1} summary:")
        print(f"    Avg train loss: {avg_epoch_loss:.4f}")
        print(f"    Epoch time: {epoch_time:.1f}s")
        print(f"    Total time: {total_time:.1f}s")
        print(f"    Tokens processed: {total_tokens:,}")
        
        # Evaluate
        if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
            train_loss = estimate_loss(model, train_data, context_length, batch_size, device)
            val_loss = estimate_loss(model, val_data, context_length, batch_size, device)
            
            if prev_val_loss is not None:
                delta = val_loss - prev_val_loss
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = "N/A"
            
            print(f"\n  === EVALUATION ===")
            print(f"    Train loss: {train_loss:.4f}")
            print(f"    Val loss: {val_loss:.4f}")
            print(f"    Delta: {delta_str}")
            print(f"  ==================")
            
            prev_val_loss = val_loss
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final val loss: {val_loss:.4f}")
    
    # Move model to CPU for saving (saves space in checkpoint)
    model = model.cpu()
    
    # Save the model
    model.save_model('workshop_model5.pt', stoi, itos, train_loss=train_loss.item(), val_loss=val_loss.item())
    
    # Generate (on CPU for compatibility)
    print("\n" + "="*60)
    print("GENERATED TEXT")
    print("="*60)
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, 500)[0].tolist()))
