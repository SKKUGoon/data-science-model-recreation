import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        """
        Initialize the self-attention layer.
        Args:
            dim (int): The dimension of the input + output. (Embedding dimension)
        """
        super().__init__()
        self.dim = dim

        # learned weights to project input X into queries (Q), keys (K), and values (V)
        # we are creaging single attention head matrix.
        # normally, the llms have multiple attention heads. they are stacked and then combined at the end.
        # but here, we are creating a single attention head matrix. Therefore the learned weights
        # do not have to shrink the dimension. We can use keep the dimension the same.
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)

        # Dropout applied to attention. 
        self.dropout = nn.Dropout(0.1)

        # Final projection
        self.w_o = nn.Linear(dim, dim)

    def forward(self, X: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Forward pass for attention.

        Args:
            X (torch.Tensor): input tensor of shape (batch size, seq_len, dim). 
            attention_mask (torch.Tensor): Binary mask of shape (batch size, seq_len, seq_len). 
            where 0 indicates the positions to be masked.
        
        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_len, dim).
        """
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        # dot-product attention scores
        # Finding similarity between Q and K
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)

        # Apply attention mask
        if attention_mask is not None:
            # Replace masked position with large negative value
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Normalize scores with softmax to make the sum of row to 1
        attention_weight = torch.softmax(scores, dim=-1) # (seq_len, seq_len)

        # Apply dropout
        attention_weight = self.dropout(attention_weight)

        # Compute the final output
        output = attention_weight @ V

        return self.w_o(output)
