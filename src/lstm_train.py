import os
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer
from src.lstm_model import LSTMNextToken
from src.next_token_dataset import make_dataloader


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip: float): # Обучение
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["input_ids"].to(device)     
        y = batch["targets"].to(device)        
        lengths = batch["lengths"].to(device)  

        optimizer.zero_grad()
        
        logits, _ = model(x, lengths=lengths)     

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
def evaluate_loss(model, loader, criterion, device): # Валидация
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


@torch.no_grad()
def calc_metrics(model, loader, tokenizer, device): # Высчитывание метрик
    model.eval()
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []

    for batch in tqdm(loader, desc="Calc Metrics", leave=False):
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        lengths = batch["lengths"].to(device)

        B = x.size(0)
        for i in range(B):
            L = int(lengths[i].item())
            if L <= 2:
                continue

            x_i = x[i, :L]
            y_i = y[i, :L]
            full = torch.cat([x_i, y_i[-1:].clone()], dim=0)
            full_len = full.size(0)

            prompt_len = max(1, int(full_len * 0.75))
            target_len = full_len - prompt_len
            
            prompt = full[:prompt_len]
            ref_tail = full[prompt_len:]

            gen_full = model.generate(
                input_ids=prompt, 
                num_new_tokens=target_len, 
                eos_id=None
            )
            
            gen_tail = gen_full[prompt_len:]

            pred_text = tokenizer.decode(gen_tail.tolist(), skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(ref_tail.tolist(), skip_special_tokens=True).strip()

            if pred_text and ref_text:
                predictions.append(pred_text)
                references.append(ref_text)

    scores = rouge.compute(predictions=predictions, references=references)
    return scores


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
        print(f"Plot saved to {save_path}")

def main(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = cfg["tokenizer"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    best_val_loss = float("inf")

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
        
        rouge_scores = calc_metrics(model, val_loader, tokenizer, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        r1 = rouge_scores.get('rouge1', 0.0)
        r2 = rouge_scores.get('rouge2', 0.0)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Rouge1: {r1:.4f} | "
            f"Rouge2: {r2:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "pad_id": pad_id,
                    "vocab_size": vocab_size,
                    "model_name": model_name,
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")

    plot_losses(train_losses, val_losses, save_path=plot_path if str(plot_path).strip() else None)


if __name__ == "__main__":
    cfg = load_yaml("configs/config.yaml")
    main(cfg)