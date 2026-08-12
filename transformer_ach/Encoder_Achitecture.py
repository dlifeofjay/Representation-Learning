import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = "cuda" if torch.cuda.is_available() else "cpu"
# Transformer Achitecture

class Head(nn.Module):
    def __init__(self, embed_dim, head_size=135, dropout=0.5):
        super().__init__()

        self.query = nn.Linear(embed_dim, head_size)
        self.key = nn.Linear(embed_dim, head_size)
        self.value = nn.Linear(embed_dim, head_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, S, T = x.shape

        q = self.query(x)
        k = self.key(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        soft_scores = F.softmax(scores, dim=-1)
        drop_scores = self.dropout(soft_scores)

        v = self.value(x)

        out = drop_scores @ v

        return out

# MultiHead Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size=135, dropout=0.5):
        super().__init__()

        self.heads = nn.ModuleList([Head(embed_dim, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))

        return out

# FeedForward

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.mlp = nn.Sequential(
        nn.Linear(embed_dim, embed_dim//2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(embed_dim//2, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Attention Block

class Block(nn.Module):
  def __init__(self, embed_dim, n_head=3, head_size=135):
    super().__init__()

    head_size = embed_dim // n_head
    self.self_attention = MultiHeadAttention(embed_dim, n_head, head_size)
    self.feed_forward = MLP(embed_dim)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, x):
    x = x + self.self_attention(self.ln1(x))
    x = x + self.feed_forward(self.ln2(x))
    return x


class FullAchitecture(nn.Module):
    def __init__(self, embed_dim, cat_col, n_layer=2, n_head=3, head_size=135):
        super().__init__()

        embed_size = 50

        self.trans_type = nn.Embedding(2, embed_size)
        self.loca = nn.Embedding(43, embed_size)
        self.merch = nn.Embedding(100, embed_size)
        self.chan = nn.Embedding(3, embed_size)
        self.custoc = nn.Embedding(4, embed_size)
        self.login = nn.Embedding(5, embed_size)
        self.day = nn.Embedding(31, embed_size)
        self.month = nn.Embedding(12, embed_size)

        self.pos = nn.Embedding(500, embed_dim)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.embedding_proj = nn.Linear(embed_dim, 120)
        self.cat_targ = nn.Linear(120, cat_col)
        self.num_col = nn.Linear(120, 1)

    def forward(self, x):
        B, S, T = x.shape

        trans_type = self.trans_type(x[:, :, 1].long())
        location = self.loca(x[:, :, 2].long())
        merchant = self.merch(x[:, :, 3].long())
        channel = self.chan(x[:, :, 4].long())
        custocu = self.custoc(x[:, :, 6].long())
        login = self.login(x[:, :, 8].long())
        day = self.day(x[:, :, 10].long())
        month = self.month(x[:, :, 11].long())
        positional_embedding = self.pos(
            torch.arange(S, device=device))

        # need to select what i need and create new matrix
        new_x = torch.cat([x[:,:,0].unsqueeze(-1), trans_type, location, merchant, channel,
                           x[:,:,5].unsqueeze(-1), custocu, x[:,:,7].unsqueeze(-1), login, x[:,:,9].unsqueeze(-1),
                           day, month, x[:,:,12].unsqueeze(-1)], dim=-1)

        x = new_x + positional_embedding

        x = self.blocks(x)
        x = self.ln_f(x)

        # Compute attention score for each transaction
        scores = self.attn_pool(x)          # 5 X 1

        # Normalize scores
        weights = torch.softmax(scores, dim=1) # 5 X 1

        # Weighted sum over transactions
        account_embedding = (weights * x).sum(dim=1)   # out 1 X embed_dim

        # Final projection
        account_embedding = self.embedding_proj(account_embedding)   # batch, 1, 120

        col_targ = self.cat_targ(account_embedding)

        num_targ = self.num_col(account_embedding)

        return account_embedding, col_targ, num_targ