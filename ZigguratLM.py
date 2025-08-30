import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import os
import math
import random
import argparse
from collections import Counter

@dataclass
class ModelConfig:
    block_size: int = 256
    n_embd: int = 384
    n_stages: int = 2
    n_layer_per_stage: int = 3
    n_head: int = 6
    dropout: float = 0.1
    num_codes: int = 1024
    vq_levels: int = 4
    commitment_cost: float = 0.25 # This is now only used for logging, not training
    vocab_size: int = None

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim, self.base = dim, base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached, self.cos_cached, self.sin_cached = None, None, None
    def _cache(self, seq_len, device):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached, self.sin_cached = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached
    def forward(self, x):
        seq_len = x.shape[1]
        cos, sin = self._cache(seq_len, device=x.device)
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rotated * sin

class SimplifiedRetention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.q_proj, self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False), nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj, self.g_proj = nn.Linear(config.n_embd, config.n_embd, bias=False), nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rope = RotaryEmbedding(dim=self.head_dim)
        self.dropout = nn.Dropout(config.dropout)
        decay = 1.0 - torch.exp2(-5.0 - torch.arange(0, config.n_head, dtype=torch.float32))
        self.decay = nn.Parameter(decay, requires_grad=False)
    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.q_proj(x).view(B,S,self.config.n_head,self.head_dim), self.k_proj(x).view(B,S,self.config.n_head,self.head_dim), self.v_proj(x).view(B,S,self.config.n_head,self.head_dim)
        q, k = self.rope(q), self.rope(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        indices = torch.arange(S, device=x.device)
        d_mask = torch.tril(indices.view(S, 1) - indices.view(1, S))
        gamma = self.decay.view(1, self.config.n_head, 1, 1)
        decay_matrix = torch.pow(gamma, d_mask) * torch.tril(torch.ones(S, S, device=x.device))
        retention = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)) * decay_matrix
        output = (retention @ v).transpose(1, 2).contiguous().view(B, S, D)
        g = F.silu(self.g_proj(x))
        return self.dropout(self.out_proj(output * g))

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding_dim, self.num_codes, self.commitment_cost = embedding_dim, num_codes, commitment_cost
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)
    def forward(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_codes).float()
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        q_latent_loss, e_latent_loss = F.mse_loss(quantized, inputs.detach()), F.mse_loss(quantized.detach(), inputs)
        quantized_out = inputs + (quantized - inputs).detach()
        return quantized_out, q_latent_loss, e_latent_loss, quantized, encoding_indices.view(inputs.shape[:-1])

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_levels: int, num_codes: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_levels = num_levels
        self.quantizers = nn.ModuleList([VectorQuantizer(num_codes, embedding_dim, commitment_cost) for _ in range(num_levels)])
    def forward(self, inputs):
        residual, total_quantized_output, all_indices = inputs, torch.zeros_like(inputs), []
        total_q_loss, total_e_loss = 0.0, 0.0
        for quantizer in self.quantizers:
            _, q_loss, e_loss, quantized_pure, indices = quantizer(residual)
            all_indices.append(indices)
            total_quantized_output += quantized_pure
            total_q_loss, total_e_loss = total_q_loss + q_loss, total_e_loss + e_loss
            residual = residual - quantized_pure.detach()
        final_quantized_ste = inputs + (total_quantized_output - inputs).detach()
        return final_quantized_ste, total_q_loss/self.num_levels, total_e_loss/self.num_levels, torch.stack(all_indices, dim=0)

class DiscretizedManifoldBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1, self.retention, self.ln_2 = nn.LayerNorm(config.n_embd), SimplifiedRetention(config), nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4*config.n_embd), nn.GELU(), nn.Linear(4*config.n_embd, config.n_embd), nn.Dropout(config.dropout))
        self.vq = ResidualVectorQuantizer(config.vq_levels, config.num_codes, config.n_embd, config.commitment_cost)
        self.ln_3 = nn.LayerNorm(config.n_embd)
    def forward(self, x, return_indices=False):
        x = x + self.retention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x_norm = self.ln_3(x)
        quantized_x, q_loss, e_loss, indices = self.vq(x_norm)
        output = x + quantized_x
        if return_indices: return output, q_loss, e_loss, indices
        return output, q_loss, e_loss

class ResidualTokenFusionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fusion_conv = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=2, stride=2)
        self.fusion_ln = nn.LayerNorm(config.n_embd)
        self.skip_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.activation = nn.GELU()
        self.final_ln = nn.LayerNorm(config.n_embd)
    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        x_fused = self.fusion_conv(x_permuted)
        x_fused = self.fusion_ln(x_fused.permute(0, 2, 1))
        x_skip = self.skip_pool(x_permuted)
        x_skip = x_skip.permute(0, 2, 1)
        output = self.activation(x_fused + x_skip)
        return self.final_ln(output)

class VulkanCompatibleEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Store embedding weights as a regular parameter
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        
        # Create a linear layer for each possible token value
        # This is inefficient in terms of memory but works around Vulkan limitations
        self.embed_layers = nn.ModuleList([
            nn.Linear(1, embedding_dim, bias=False) for _ in range(num_embeddings)
        ])
        
        # Initialize the weights of each linear layer to match the corresponding embedding
        for i, layer in enumerate(self.embed_layers):
            layer.weight.data = self.weight[i:i+1].clone()
    
    def forward(self, indices):
        batch_size, seq_len = indices.shape
        device = indices.device
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
        
        # For each position in the sequence, get the corresponding embedding
        for i in range(batch_size):
            for j in range(seq_len):
                token_idx = int(indices[i, j].item())
                # Create a dummy input for the linear layer
                dummy_input = torch.ones(1, 1, device=device)
                # Get the embedding using the corresponding linear layer
                embedding = self.embed_layers[token_idx](dummy_input)
                embeddings[i, j] = embedding.squeeze(0)
        
        return embeddings
    
    def sync_weights(self):
        # Sync the weights from the main weight parameter to the linear layers
        for i, layer in enumerate(self.embed_layers):
            layer.weight.data = self.weight[i:i+1].clone()

class DiscretizedManifoldTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Use our custom Vulkan-compatible embedding
        self.wte = VulkanCompatibleEmbedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.stages, self.fusion_blocks = nn.ModuleList(), nn.ModuleList()
        for i in range(config.n_stages):
            self.stages.append(nn.ModuleList([DiscretizedManifoldBlock(config) for _ in range(config.n_layer_per_stage)]))
            if i < config.n_stages - 1: self.fusion_blocks.append(ResidualTokenFusionBlock(config))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tie weights with our custom embedding
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None: torch.nn.init.zeros_(module.bias)
        total_layers = self.config.n_stages * self.config.n_layer_per_stage
        if isinstance(module, SimplifiedRetention): torch.nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2*total_layers))
        if hasattr(module, 'mlp') and isinstance(module.mlp[-2], nn.Linear): torch.nn.init.normal_(module.mlp[-2].weight, mean=0.0, std=0.02/math.sqrt(2*total_layers))
    def forward(self, idx, targets=None, return_indices=False):
        # idx is already a float tensor for Vulkan compatibility
        # Our custom embedding layer will handle the conversion internally
        
        # Sync embedding weights before forward pass
        self.wte.sync_weights()
        
        x = self.drop(self.wte(idx))
        all_stage_indices, output_from_stage1 = [], None
        total_q_loss, total_e_loss = 0.0, 0.0
        for i, stage in enumerate(self.stages):
            stage_indices = []
            for block in stage:
                if return_indices:
                    x, q_loss, e_loss, indices = block(x, return_indices=True)
                    stage_indices.append(indices)
                else:
                    x, q_loss, e_loss = block(x)
                total_q_loss, total_e_loss = total_q_loss + q_loss, total_e_loss + e_loss
            if return_indices: all_stage_indices.append(torch.stack(stage_indices))
            if i == 0: output_from_stage1 = x
            if i < self.config.n_stages - 1:
                if x.shape[1] > 1: x = self.fusion_blocks[i](x)
                else: break
        logits = self.lm_head(self.ln_f(output_from_stage1))
        if return_indices: return logits, all_stage_indices
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            total_layers = self.config.n_stages * self.config.n_layer_per_stage
            return logits, (ce_loss, total_q_loss / total_layers, total_e_loss / total_layers)
        return logits, None

@torch.no_grad()
def display_bucket_contents(model, tokenizer, device, analysis_tokens, stage_to_show=0, layer_to_show=0, top_k=8):
    model.eval()
    print(f"\n--- BUCKET ANALYSIS (Stage {stage_to_show}, Layer {layer_to_show}) ---")
    # Convert to float for device compatibility
    tokens = analysis_tokens.float().unsqueeze(0).to(device)
    _, all_stage_indices = model(tokens, return_indices=True)
    config = model.config
    seq_len_multiplier = (0.5 ** stage_to_show)
    if stage_to_show >= len(all_stage_indices):
        print(f"Error: Stage {stage_to_show} not found in model output."); model.train(); return
    layer_indices = all_stage_indices[stage_to_show][layer_to_show]
    if stage_to_show > 0:
        print("--- Analyzing Multi-Token Concepts (Token labels not applicable) ---")
        for level_idx in range(config.vq_levels):
            level_counter = Counter(layer_indices[level_idx].flatten().tolist())
            print(f"\n--- VQ HIERARCHY LEVEL {level_idx} ---")
            print(f"  Top 5 most used buckets: {level_counter.most_common(5)}")
    else:
        bucket_contents = [[Counter() for _ in range(config.num_codes)] for _ in range(config.vq_levels)]
        expected_seq_len = int(config.block_size * seq_len_multiplier)
        for level_idx in range(config.vq_levels):
            for seq_idx in range(expected_seq_len):
                token_id = tokens[0, seq_idx].item()
                bucket_id = layer_indices[level_idx, 0, seq_idx].item()
                try: bucket_contents[level_idx][bucket_id][tokenizer.decode([int(token_id)])] += 1
                except: continue
        for level_idx in range(config.vq_levels):
            print(f"\n--- VQ HIERARCHY LEVEL {level_idx} ---")
            used_buckets = sorted([(i, c) for i, c in enumerate(bucket_contents[level_idx]) if c], key=lambda item: len(item[1]), reverse=True)
            for bucket_id, counter in used_buckets[:5]: print(f"  Bucket [{bucket_id:4d}] (Unique: {len(counter):,d}): {counter.most_common(top_k)}")
    print("--- End of Bucket Analysis ---\n")
    model.train()

@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens=50):
    model.eval()
    # Use float32 for token tensors to support Vulkan
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.float, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = prompt_tokens[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits, probs = logits[:, -1, :], F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        prompt_tokens = torch.cat((prompt_tokens, next_token.float()), dim=1)
    decoded = tokenizer.decode([int(t) for t in prompt_tokens.squeeze().tolist()])
    print(f"--- PROMPT: '{prompt}' ---\nGENERATION: {decoded}\n-------------------------------------")
    model.train()

def chat(checkpoint_path, device, max_new_tokens=256):
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Add ModelConfig to safe globals for loading
    torch.serialization.add_safe_globals([ModelConfig])
    
    # Load checkpoint to CPU first to avoid device mismatch issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    tokenizer = tiktoken.get_encoding("cl100k_base")
    config.vocab_size = tokenizer.n_vocab
    model = DiscretizedManifoldTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nModel loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params). Chat away!")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit']: break
        # Use float32 for token tensors to support Vulkan
        prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.float, device=device).unsqueeze(0)
        print("Model:", end=" ", flush=True)
        for _ in range(max_new_tokens):
            idx_cond = prompt_tokens[:, -model.config.block_size:]
            with torch.no_grad(): 
                # Sync embedding weights before forward pass
                model.wte.sync_weights()
                logits, _ = model(idx_cond)
            logits, probs = logits[:, -1, :], F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if hasattr(tokenizer, 'eot_token') and next_token.item() == tokenizer.eot_token: break
            prompt_tokens = torch.cat((prompt_tokens, next_token.float()), dim=1)
            print(tokenizer.decode([int(next_token.item())]), end="", flush=True)
        print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or chat with a Multi-Stage Hierarchical DMT.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint for chat mode.')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from.')
    args = parser.parse_args()
    
    # Try to use Vulkan if available, otherwise CPU
    if torch.is_vulkan_available():
        device = 'vulkan'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    if args.checkpoint:
        chat(args.checkpoint, device)
    else:
        print("Starting training...")
        NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = 5, 8, 3e-4
        EVAL_EVERY = 500
        CHECKPOINT_DIR = "checkpoints"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()
        # Use float32 for token tensors to support Vulkan
        all_data = torch.tensor(tokenizer.encode(text), dtype=torch.float)
        n = int(0.9 * len(all_data))
        train_data, val_data = all_data[:n], all_data[n:]
        
        # Initialize variables for resuming
        start_epoch = 0
        global_step = 0
        
        if args.resume:
            print(f"Resuming training from checkpoint: {args.resume}")
            # Add ModelConfig to safe globals for loading
            torch.serialization.add_safe_globals([ModelConfig])
            
            # Load checkpoint to CPU first to avoid device mismatch issues
            checkpoint = torch.load(args.resume, map_location='cpu')
            config = checkpoint['config']
            model = DiscretizedManifoldTransformer(config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create optimizers
            main_params = [p for name, p in model.named_parameters() if 'vq.quantizers' not in name]
            optimizer_main = torch.optim.AdamW(main_params, lr=LEARNING_RATE, betas=(0.9, 0.95))
            vq_params = [p for name, p in model.named_parameters() if 'vq.quantizers' in name]
            optimizer_vq = torch.optim.AdamW(vq_params, lr=LEARNING_RATE / 10.0, betas=(0.9, 0.95))
            
            # Try to load optimizer states if they exist
            if 'optimizer_main_state_dict' in checkpoint and 'optimizer_vq_state_dict' in checkpoint:
                optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
                optimizer_vq.load_state_dict(checkpoint['optimizer_vq_state_dict'])
                print("Loaded optimizer states from checkpoint.")
            else:
                print("Warning: Optimizer states not found in checkpoint. Using new optimizers.")
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            print(f"Resuming from epoch {start_epoch}, global step {global_step}")
        else:
            config = ModelConfig()
            config.vocab_size = tokenizer.n_vocab
            model = DiscretizedManifoldTransformer(config).to(device)
            
            # Create optimizers
            main_params = [p for name, p in model.named_parameters() if 'vq.quantizers' not in name]
            optimizer_main = torch.optim.AdamW(main_params, lr=LEARNING_RATE, betas=(0.9, 0.95))
            vq_params = [p for name, p in model.named_parameters() if 'vq.quantizers' in name]
            optimizer_vq = torch.optim.AdamW(vq_params, lr=LEARNING_RATE / 10.0, betas=(0.9, 0.95))
        
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
        print(f"Training with two optimizers: Main (LR: {LEARNING_RATE}) and VQ (LR: {LEARNING_RATE / 10.0}).")
        
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n--- Starting Epoch {epoch+1}/{NUM_EPOCHS} ---")
            possible_starts = list(range(len(train_data) - config.block_size - 1))
            random.shuffle(possible_starts)
            
            for i in range(len(possible_starts) // BATCH_SIZE):
                start_indices = possible_starts[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # Use float32 for token tensors to support Vulkan
                xb = torch.stack([train_data[j:j+config.block_size] for j in start_indices]).to(device)
                yb = torch.stack([train_data[j+1:j+config.block_size+1] for j in start_indices]).to(device)
                logits, (ce_loss, q_loss, e_loss) = model(xb, yb)
                
                main_loss = ce_loss
                vq_optimizer_loss = q_loss
                optimizer_main.zero_grad(set_to_none=True)
                optimizer_vq.zero_grad(set_to_none=True)
                main_loss.backward(retain_graph=True)
                vq_optimizer_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer_main.step()
                optimizer_vq.step()
                
                if global_step % 100 == 0:
                    total_loss = main_loss + vq_optimizer_loss # For display purposes only
                    print(f"Epoch {epoch+1}, Step {global_step}: Loss {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, Q_loss: {q_loss.item():.4f}, E_loss: {e_loss.item():.4f})")
                
                if global_step > 0 and global_step % EVAL_EVERY == 0:
                    generate(model, tokenizer, device, prompt="Чайник это ")
                    analysis_sample = val_data[:config.block_size]
                    display_bucket_contents(model, tokenizer, device, analysis_sample, stage_to_show=0, layer_to_show=0)
                    if config.n_stages > 1:
                        display_bucket_contents(model, tokenizer, device, analysis_sample, stage_to_show=1, layer_to_show=0)
                
                global_step += 1
                
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"dmt_checkpoint_epoch_{epoch+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch+1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_main_state_dict': optimizer_main.state_dict(),
                'optimizer_vq_state_dict': optimizer_vq.state_dict(),
                'config': config
            }, checkpoint_path)
        print("\n--- Training Finished ---")
