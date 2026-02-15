# text-autocomplete/src/eval_lstm.py

from dataclasses import dataclass

import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer

from src.lstm_model import LSTMNextToken
from src.next_token_dataset import make_dataloader


@dataclass
class Config:
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

    # генерация для ROUGE
    temperature: float = 1.0
    top_k: int | None = 50


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["input_ids"].to(device)     # [B, T]
        y = batch["targets"].to(device)       # [B, T]
        lengths = batch["lengths"].to(device) # [B]

        optimizer.zero_grad()
        logits = model(x, lengths=lengths)    # [B, T, V]

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
def eval_loss(model, loader, criterion, device):
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


@torch.no_grad()
def eval_rouge_autocomplete_3_4(
    model: LSTMNextToken,
    loader,
    tokenizer,
    device,
    max_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    """
    Сценарий:
      - восстанавливаем "исходную" последовательность токенов из (X, Y)
      - берём первые 3/4 как prompt
      - генерируем оставшиеся 1/4 (ровно столько токенов)
      - сравниваем декодированный tail с референсом через ROUGE
    """
    model.eval()
    rouge = evaluate.load("rouge")

    predictions: list[str] = []
    references: list[str] = []

    for batch in tqdm(loader, desc="ROUGE eval", leave=False):
        x = batch["input_ids"].to(device)   # [B, T]
        y = batch["targets"].to(device)     # [B, T]
        lengths = batch["lengths"].to(device)

        B = x.size(0)
        for i in range(B):
            L = int(lengths[i].item())
            if L <= 1:
                continue

            x_i = x[i, :L]      # [L]
            y_i = y[i, :L]      # [L]

            # восстановим полную последовательность (длина L+1):
            # full = [t0..t(L-1)] + [tL]
            full = torch.cat([x_i, y_i[-1:].clone()], dim=0)
            full_len = full.size(0)

            prompt_len = max(1, int(full_len * 0.75))
            if prompt_len >= full_len:
                continue

            prompt = full[:prompt_len]
            ref_tail = full[prompt_len:]
            num_new = int(ref_tail.size(0))

            gen_full = model.generate(
                input_ids=prompt,
                num_new_tokens=num_new,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                eos_id=None,  # можно поставить tokenizer.sep_token_id, но здесь генерим фиксированную длину
            )

            gen_tail = gen_full[prompt_len:]
            pred_text = tokenizer.decode(gen_tail.tolist(), skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(ref_tail.tolist(), skip_special_tokens=True).strip()

            predictions.append(pred_text)
            references.append(ref_text)

    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    results = rouge.compute(predictions=predictions, references=references)
    return results


def main(cfg: Config):
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
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_loss(model, val_loader, criterion, device)

        rouge_scores = eval_rouge_autocomplete_3_4(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_len=cfg.max_len,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | "
            f"ROUGE-1 = {rouge_scores['rouge1']:.4f} | "
            f"ROUGE-2 = {rouge_scores['rouge2']:.4f} | "
            f"ROUGE-L = {rouge_scores['rougeL']:.4f}"
        )


if __name__ == "__main__":
    main(Config())
