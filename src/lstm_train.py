import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.lstm_model import LSTMNextToken
from src.next_token_dataset import make_dataloader


@dataclass
class TrainConfig:
    model_name: str = "bert-base-uncased"
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"

    batch_size: int = 64
    max_len: int = 128

    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1

    lr: float = 3e-3
    epochs: int = 10
    grad_clip: float = 1.0

    save_dir: str = "models"
    save_name: str = "lstm_next_token.pt"


def train_epoch(model, loader, optimizer, criterion, device, pad_id: int):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["input_ids"].to(device)     # [B, T]
        y = batch["targets"].to(device)       # [B, T]
        lengths = batch["lengths"].to(device) # [B]

        optimizer.zero_grad()

        logits = model(x, lengths=lengths)    # [B, T, V]

        # CrossEntropyLoss ожидает [N, C] и [N]
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        lengths = batch["lengths"].to(device)

        logits = model(x, lengths=lengths)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def main(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    _, train_loader = make_dataloader(
        cfg.train_csv, batch_size=cfg.batch_size, shuffle=True, max_len=cfg.max_len, pad_id=pad_id
    )
    _, val_loader = make_dataloader(
        cfg.val_csv, batch_size=cfg.batch_size, shuffle=False, max_len=cfg.max_len, pad_id=pad_id
    )

    model = LSTMNextToken(
        vocab_size=vocab_size,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_id=pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # игнорируем паддинг в targets
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_val = float("inf")
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, pad_id)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "pad_id": pad_id,
                    "vocab_size": vocab_size,
                    "model_name": cfg.model_name,
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    main(TrainConfig())
