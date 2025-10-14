import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, per_head_dim: int):
        """
        Args:
            dim (int): The dimension of the input + output. (Embedding dimension)
            n_heads (int): The number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = per_head_dim
        self.total_projected_dim = n_heads * per_head_dim

        # Linear projections for queries, keys and values - learned weights
        self.w_q = nn.Linear(dim, self.total_projected_dim, bias=False)
        self.w_k = nn.Linear(dim, self.total_projected_dim, bias=False)
        self.w_v = nn.Linear(dim, self.total_projected_dim, bias=False)

        # Dropout applied to attention
        self.dropout = nn.Dropout(0.1)

        # Final projection
        self.w_o = nn.Linear(self.total_projected_dim, dim, bias=False)
    
    def forward(self, X: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.

        Args:
            X (torch.Tensor): input tensor of shape (batch size, seq_len, embeddidion dimension).
            attention_mask (torch.Tensor): Binary mask of shape (batch size, seq_len, seq_len).
                where 0 indicates the positions to be masked.
        
        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_len, dim).
        """
        B, L, _ = X.shape

        Q = self.w_q(X)  # B, L, total_projected_dim
        K = self.w_k(X)
        V = self.w_v(X)

        # Reshape into multiple heads (B)
        Q = (Q.view(B, L, self.n_heads, self.head_dim)  # Split the embedding into (self.n_heads * self.head_dim)
             .permute(0, 2, 1, 3))  # reordering the axes to have (B, self.n_heads, L, self.head_dim)
        K = (K.view(B, L, self.n_heads, self.head_dim)
             .permute(0, 2, 1, 3))
        V = (V.view(B, L, self.n_heads, self.head_dim)
             .permute(0, 2, 1, 3))

        # Compute attention scores
        # Q @ K^T -> (B, self.n_heads, L, L). L*L attention pattern for self.n_heads heads.
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e20)
        
        # Softmax to get attention weights
        scores = torch.softmax(scores, dim=-1)
        
        scores = self.dropout(scores)

        context = scores @ V
        # Reshape back to (B, L, total_projected_dim)
        context = (context.transpose(1, 2)
                   .contiguous()  # torch stores array(buffer) and indices(shape and stride) to mark matrices. 
                   #If operation like .view or .permute is performed, data is not contiguous. 
                   # .view() assume that data is laid out contiguously in memory.
                   # contiguous matches the array layout to match the indices. 
                   .view(B, L, self.total_projected_dim)
        )

        return self.w_o(context)