import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from src.lstm_model import LSTMNextToken
from src.next_token_dataset import make_dataloader
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip: float):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["input_ids"].to(device)      # [B, T]
        y = batch["targets"].to(device)        # [B, T]
        lengths = batch["lengths"].to(device)  # [B]

        optimizer.zero_grad()
        logits, _ = model(x, lengths=lengths)     # [B, T, V]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        lengths = batch["lengths"].to(device)

        logits, _ = model(x, lengths=lengths)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def plot_losses(train_losses, val_losses, save_path: str | None = None):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


def main(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["tokenizer"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_id = tokenizer.eos_token_id
    
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]

    batch_size = int(cfg["train"]["batch_size"])
    max_len = int(cfg["train"]["max_len"])

    _, train_loader = make_dataloader(
        train_csv, batch_size=batch_size, shuffle=True, max_len=max_len, pad_id=pad_id
    )
    _, val_loader = make_dataloader(
        val_csv, batch_size=batch_size, shuffle=False, max_len=max_len, pad_id=pad_id
    )

    model_cfg = cfg["model"]
    model = LSTMNextToken(
        vocab_size=vocab_size,
        emb_dim=int(model_cfg["emb_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        pad_id=pad_id,
        max_len=max_len,
    ).to(device)

    lr = float(cfg["train"]["lr"])
    epochs = int(cfg["train"]["epochs"])
    grad_clip = float(cfg["train"]["grad_clip"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    save_dir = cfg["save"]["save_dir"]
    save_name = cfg["save"]["save_name"]
    plot_path = cfg["save"]["plot_path"]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,          # сохраняем ВЕСЬ yaml как dict
                    "pad_id": pad_id,
                    "vocab_size": vocab_size,
                    "model_name": model_name,
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")

    plot_losses(train_losses, val_losses, save_path=plot_path if str(plot_path).strip() else None)



if __name__ == "__main__":
    cfg = load_yaml("configs/config.yaml")
    main(cfg)

