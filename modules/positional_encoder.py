import torch
import torch.nn as nn
import numpy as np

class RelativePositionalEncoder(nn.Module):
    """This is a relative position encoder. Actually, it is an nn.Embedding layer.
    `0` represents on the left of the <mask>;
    `1` represents the postion of the <mask>;
    `2` represents on the right of the <mask>;
    """
    def __init__(self, d_model, dropout: float = 0.1):
        # self.dropout = nn.Dropout(p=dropout)
        super().__init__()
        self.pe = nn.Embedding(num_embeddings=4, embedding_dim=d_model, padding_idx=3)
    
    def forward(self, postion_ids) -> torch.Tensor:
        """
        Args:
            postion: torch.Tensor, shape (batch_size, seq_len)
        """
        return self.pe(postion_ids)

class SinusoidalPostionalEncoder(nn.Module):
    """The sinusoidal postional encoder from `attention is all you need`
    """
    def __init__(self, max_seq_len ,d_model, dropout: float = 0.1):
        # self.dropout = nn.Dropout(p=dropout)
        super().__init__()
        self.pe = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model)
        self.create_sinusoidal_embeddings(max_seq_len, d_model, self.pe.weight)
    
    def forward(self, position_ids) -> torch.Tensor:
        return self.pe(position_ids)
    
    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
        out.requires_grad = False   # not update during training
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()