import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMNextToken(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
        max_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None, hidden=None):
        
        emb = self.embedding(input_ids)  

        if lengths is not None:
            packed = pack_padded_sequence(
                emb,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, hidden = self.lstm(packed, hidden)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  
        else:
            out, hidden = self.lstm(emb, hidden)

        out = self.dropout(out)
        logits = self.fc(out)  
        
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_new_tokens: int = 20,
        eos_id: int | None = None,
    ) -> torch.Tensor:

        was_1d = (input_ids.dim() == 1)
        if was_1d:
            input_ids = input_ids.unsqueeze(0)

        device = next(self.parameters()).device
        seq = input_ids.to(device)

        self.eval()
        _, hidden = self(seq)
        last_token = seq[:, -1:] # Последний токен промпта — это первый вход для генерации

        for _ in range(num_new_tokens):
            logits, hidden = self(last_token, hidden=hidden) 
            next_logits = logits[:, -1, :]  # Берем логиты последнего шага
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True) 
            seq = torch.cat([seq, next_token], dim=1)  # Добавляем в общую последовательность
            last_token = next_token
            if eos_id is not None:
                if (next_token.squeeze(1) == eos_id).all():
                    break

        return seq.squeeze(0) if was_1d else seq