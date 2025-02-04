import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
@dataclass
class PICOBERTConfig:
    vocab_size: int = 100000
    n_embd : int = 768
    n_layers: int = 12
    n_hidden_layer_size: int = 768
    n_attention_heads: int = 12
    attn_head_size: int = 64
    type_vocab_size: int = 2
    max_pos_embd : int = 512
    dropout: float = 0.15
    device = "cuda" if torch.cuda.is_available() else "cpu"


class BertEmbedding(nn.Module):
    def __init__(self, config: PICOBERTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.device = config.device
        self.segment_embedding = nn.Embedding(config.type_vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_pos_embd, config.n_embd)

    def forward(self, input_ids, token_type_ids):
        position_ids = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        seg_emb = self.segment_embedding(token_type_ids)
        pos_emb = self.position_embedding(position_ids)
        embeddings = tok_emb + seg_emb + pos_emb
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config: PICOBERTConfig):
        super().__init__()
        assert config.n_embd % config.n_attention_heads == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.PICOBERT_SCALE_INIT = 1
        self.n_attn_heads = config.n_attention_heads
        self.attn_head_size = config.attn_head_size
        self.n_embd = config.n_embd
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_attn_heads, C // self.n_attn_heads).transpose(1, 2)
        k = k.view(B, T, self.n_attn_heads, self.attn_head_size).transpose(1, 2)
        v = v.view(B, T, self.n_attn_heads, self.attn_head_size).transpose(1, 2)
        att = q @ k.transpose(-2, -1) * self.attn_head_size ** -0.5
        att = att + attention_mask
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FFN(nn.Module):
    def __init__(self, config: PICOBERTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_hidden_layer_size)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_hidden_layer_size, config.n_embd)
        self.c_proj.PICOBERT_SCALE_INIT = 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: PICOBERTConfig):
        super().__init__()
        self.attn = BertSelfAttention(config)
        self.ffn = FFN(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, extended_mask):
        x = x + self.attn(self.ln1(x), extended_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class PICOBERT(nn.Module):
    def __init__(self, config: PICOBERTConfig):
        super().__init__()
        self.config = config
        self.bert = nn.ModuleDict(dict(
            bert_embedding=BertEmbedding(config),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.mlm_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.n_embd, config.vocab_size)
        )
        self.nsp_head = nn.Linear(config.n_embd, 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "PICOBERT_SCALE_INIT"):
                std *= (self.config.n_hidden_layer_size) ** -0.5
            torch.nn.init.normal_(module.weight, 0, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0, std)

    def forward(self, idx, attention_mask, token_type_ids, mlm_targets=None, nsp_targets=None):

        extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        x = self.bert.bert_embedding(idx, token_type_ids)
        for block in self.bert.h:
            x = block(x, extended_mask)
        outputs = self.bert.ln_f(x)

        mlm_logits = self.mlm_head(outputs)
        nsp_logits = self.nsp_head(outputs[:, 0])

        loss = None
        if mlm_targets is not None and nsp_targets is not None:
            mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            nsp_loss_fn = nn.CrossEntropyLoss()
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, self.config.vocab_size), mlm_targets.view(-1))
            nsp_loss = nsp_loss_fn(nsp_logits, nsp_targets)
            loss = mlm_loss + nsp_loss

        return mlm_logits, nsp_logits, loss
