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
        """
        Возвращает (logits, hidden_state)
        """
        emb = self.embedding(input_ids)  # [B, T, E]

        if lengths is not None:
            # Режим обучения (с использованием pack_padded_sequence)
            packed = pack_padded_sequence(
                emb,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            # hidden можно передать, даже если используем packed
            packed_out, hidden = self.lstm(packed, hidden)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H]
        else:
            # Режим генерации или обычный проход
            out, hidden = self.lstm(emb, hidden)  # [B, T, H]

        out = self.dropout(out)
        logits = self.fc(out)  # [B, T, V]
        
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

        # 1. "Прогреваем" LSTM на всем промпте, чтобы получить начальный hidden state
        # Нам не нужны логиты промпта, только скрытое состояние после него
        _, hidden = self(seq)
        
        # Последний токен промпта — это первый вход для генерации
        last_token = seq[:, -1:] # [B, 1]

        for _ in range(num_new_tokens):
            # 2. Подаем только один токен и прошлое состояние
            # lengths не нужен, так как длина фиксирована = 1
            logits, hidden = self(last_token, hidden=hidden) # logits: [B, 1, V]
            
            # Берем логиты последнего (единственного) шага
            next_logits = logits[:, -1, :]        # [B, V]
            
            # Жадная стратегия (argmax)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True) # [B, 1]

            # Добавляем в общую последовательность
            seq = torch.cat([seq, next_token], dim=1)

            # Обновляем вход для следующего шага
            last_token = next_token

            if eos_id is not None:
                # Если сгенерировали EOS (учитываем, что у нас может быть батч)
                # Для простоты прерываем, если ВСЕ в батче закончили, или игнорируем для батча > 1
                if (next_token.squeeze(1) == eos_id).all():
                    break

        return seq.squeeze(0) if was_1d else seq