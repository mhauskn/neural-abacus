import torch
import torch.nn as nn
from torch.nn import functional as F
from abacus import Soroban
from typing import Union
import numpy as np
from gpt import Block

EMBED_DIM = 64
ABACUS_ROWS = 7
ABACUS_COLS = 13
MAX_TEMPORAL = 40

class Config:
    """ a lightweight configuration class inspired by yacs """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Model(nn.Module):
    """Model to manipulate the abacus. """
    
    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.op_embed = nn.Embedding(num_embeddings=3, embedding_dim=EMBED_DIM)
        self.numerical_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBED_DIM)
        self.position_embed = nn.Embedding(num_embeddings=ABACUS_COLS, embedding_dim=EMBED_DIM)
        self.temporal_embed = nn.Embedding(num_embeddings=MAX_TEMPORAL, embedding_dim=EMBED_DIM)

        self.ln1 = nn.LayerNorm(4*EMBED_DIM)

        # Embeddings used for the previous instructions
        self.fn_embed = nn.Embedding(num_embeddings=5, embedding_dim=EMBED_DIM)
        self.col_embed = nn.Embedding(num_embeddings=ABACUS_COLS, embedding_dim=EMBED_DIM)
        self.n_embed = nn.Embedding(num_embeddings=5, embedding_dim=EMBED_DIM)

        self.conv1 = nn.Conv1d(in_channels=ABACUS_ROWS, out_channels=EMBED_DIM, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=3*EMBED_DIM, out_channels=EMBED_DIM, kernel_size=2, stride=1)

        # Transformer block for attention over abacus columns
        self.col_block = Block(Config(
            block_size=64,
            n_embd=EMBED_DIM,
            n_head=8,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            use_temporal_attn_mask=False,
        ))

        # Transformer block for operations over time
        self.block = Block(Config(
            block_size=64,
            n_embd=4*EMBED_DIM,
            n_head=8,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            use_temporal_attn_mask=True,
        ))

        self.output_fn = nn.Linear(in_features=4*EMBED_DIM, out_features=5)
        self.output_column = nn.Linear(in_features=4*EMBED_DIM, out_features=ABACUS_COLS)
        self.output_n = nn.Linear(in_features=4*EMBED_DIM, out_features=5)

    def forward(self, ab_reps: torch.Tensor, numerical_inputs: torch.Tensor, prev_instrs: torch.Tensor,
        ops: torch.Tensor, targets=None):
        """Does one step of manipulation to the abacus.
        
        ab_reps: (Batch x t x 7 x 13) abacus representations.
        numerical_inputs: (Batch x 13) tensor containing the numerical value.
        prev_instrs: (Batch x t x 3) tensor containing the previously issued instructions.
        ops: (Batch x 1) tensor containing the operation to perform.
        targets: (Batch x 3) of actions to perform.

        Returns loss if targets were provided.

        """
        device = ab_reps.device
        b, t, rows, cols = ab_reps.size()
        num_emb = self.numerical_embed(numerical_inputs)
        pos_emb = self.position_embed(torch.arange(0, cols, dtype=torch.long, device=device).unsqueeze(0))
        target_val_enc = num_emb + pos_emb
        # Add a repeated time dimension to the target val encoding.
        target_val_enc = torch.repeat_interleave(target_val_enc, t, dim=0)

        # Embed the operation
        op_enc = self.op_embed(ops) # --> <batch, 1, embed_dim>
        op_enc = torch.repeat_interleave(op_enc, cols, dim=1) # --> <batch, n_cols, embed_dim>
        op_enc = torch.repeat_interleave(op_enc, t, dim=0) # --> <batch x time, n_cols, embed_dim>

        # Combine the batch and time dimensions into a superbatch
        ab_enc = ab_reps.flatten(start_dim=0, end_dim=1) # <batch x time, 7, 13>
        # Then conv1d to embed the abacus-specific encoding
        ab_enc = self.conv1(ab_enc) # --> <batch x time, 64, 13>
        ab_enc = ab_enc.permute(0,2,1) + pos_emb # --> <batch x time, n_cols, embed_dim>

        # Fuse the target values with the abacus representations along the embedding dimension.
        x = torch.cat([target_val_enc, op_enc, ab_enc], dim=-1) 
        x = F.gelu(x)

        # Joint convolution of abacus and targets to reduce dim back to 64
        x = self.conv2(x.permute(0,2,1)) # --> (batch x time, embed_dim, n_cols)
        x = F.gelu(x)

        # Apply the transformer block over the abacus columns
        x = self.col_block(x.permute(0,2,1)) # --> (batch x time, n_cols, embed_dim)

        # Aggregate across abacus columns - choices are Sum/Last/Max.
        x = x.sum(dim=1) # --> <batch x time, embed_dim>

        # Separate the batch and time dimensions
        x = x.view(b, t, -1) # --> <batch, time, embed_dim>

        # Add the temporal embedding
        time_emb = self.temporal_embed(
            torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0))
        x = x + time_emb

        # Embed the previous instructions
        fn = self.fn_embed(prev_instrs[:,:,0])
        col = self.col_embed(prev_instrs[:,:,1])
        n = self.n_embed(prev_instrs[:,:,2])
        x = torch.cat([x, fn, col, n], dim=-1)

        # Apply the transformer block
        x = self.block(x) # <batch, time, embed_dim>
        x = self.ln1(x)

        fn_logits = self.output_fn(x)
        col_logits = self.output_column(x)
        n_logits = self.output_n(x)

        loss = None
        if targets is not None:
            fn_targets = targets[:,:,0]
            col_targets = targets[:,:,1]
            n_targets = targets[:,:,2]

            fn_loss = F.cross_entropy(fn_logits.view(-1, fn_logits.size(-1)), fn_targets.view(-1), ignore_index=-1)
            col_loss = F.cross_entropy(col_logits.view(-1, col_logits.size(-1)), col_targets.view(-1), ignore_index=-1)
            n_loss = F.cross_entropy(n_logits.view(-1, n_logits.size(-1)), n_targets.view(-1), ignore_index=-1)
            loss = fn_loss + col_loss + n_loss

        return (fn_logits, col_logits, n_logits), loss


    def decode(self, ab_reps: torch.Tensor, numerical_inputs: torch.Tensor, prev_instrs: torch.Tensor, 
        ops: torch.Tensor, do_sample=False):
        """Decodes a single step of output."""
        def sample(logits):
            # if logits.ndim == 3: # Take the most recent timestep
            #     logits = logits[:,-1,:]
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                y = torch.multinomial(probs, num_samples=1)
            else:
                _, y = torch.topk(probs, k=1, dim=-1)
            return y
        (fn_logits, col_logits, n_logits), _ = self.forward(ab_reps, numerical_inputs, prev_instrs, ops)
        fn_idx = sample(fn_logits)
        col_idx = sample(col_logits)
        n_idx = sample(n_logits)
        return torch.cat([fn_idx, col_idx, n_idx], dim=-1)
