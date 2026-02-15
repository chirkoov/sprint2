# text-autocomplete/src/lstm_model.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMNextToken(nn.Module):
    """
    Language model: по входной последовательности токенов предсказывает следующий токен
    на каждом шаге (т.е. возвращает logits для каждого положения).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

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

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids: LongTensor [B, T]  (X)
        lengths:   LongTensor [B]     (длины без padding), опционально

        Returns:
            logits: FloatTensor [B, T, vocab_size]
            logits[:, t] -> распределение для предсказания следующего токена после позиции t
        """
        emb = self.embedding(input_ids)  # [B, T, E]

        if lengths is not None:
            packed = pack_padded_sequence(
                emb,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H]
        else:
            out, _ = self.lstm(emb)  # [B, T, H]

        out = self.dropout(out)
        logits = self.fc(out)  # [B, T, V]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_new_tokens: int = 20,
        max_len: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """
        Авторегрессивная генерация: добавляет num_new_tokens токенов к prompt.

        Args:
            input_ids: LongTensor [B, T] или [T]
            num_new_tokens: сколько новых токенов сгенерировать
            max_len: если последовательность длиннее, используем только последние max_len токенов
            temperature: >0, при 1.0 без изменений; <1.0 более "уверенно", >1.0 более "случайно"
            top_k: если задано, сэмплируем только из top_k токенов (иначе сэмпл по всему словарю)
            eos_id: если задано, останавливаемся когда eos_id сгенерирован для всех элементов батча

        Returns:
            LongTensor [B, T + num_generated]
        """
        was_1d = (input_ids.dim() == 1)
        if was_1d:
            input_ids = input_ids.unsqueeze(0)

        device = next(self.parameters()).device
        seq = input_ids.to(device)

        self.eval()

        for _ in range(num_new_tokens):
            ctx = seq[:, -max_len:] if seq.size(1) > max_len else seq
            lengths = torch.full((ctx.size(0),), ctx.size(1), dtype=torch.long, device=device)

            logits = self(ctx, lengths=lengths)          # [B, T, V]
            next_logits = logits[:, -1, :]               # [B, V]

            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None and top_k > 0:
                vals, idxs = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
                probs = torch.softmax(vals, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)      # [B, 1] in 0..k-1
                next_token = idxs.gather(-1, sampled)                  # [B, 1] in vocab ids
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)   # [B, 1]

            seq = torch.cat([seq, next_token], dim=1)                  # [B, T+1]

            if eos_id is not None:
                if (next_token.squeeze(1) == eos_id).all():
                    break

        return seq.squeeze(0) if was_1d else seq
