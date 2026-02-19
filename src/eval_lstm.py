import os
import yaml
import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer

from src.lstm_model import LSTMNextToken
from src.next_token_dataset import make_dataloader


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    """Считает только средний Loss"""
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Evaluating Loss", leave=False):
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
def eval_rouge_autocomplete_3_4(model, loader, tokenizer, device):
    """
    Генерирует продолжение для последних 25% текста и считает ROUGE.
    """
    model.eval()
    rouge = evaluate.load("rouge")

    predictions: list[str] = []
    references: list[str] = []

    for batch in tqdm(loader, desc="Evaluating ROUGE", leave=True):
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        lengths = batch["lengths"].to(device)

        B = x.size(0)
        for i in range(B):
            L = int(lengths[i].item())
            if L <= 2:
                continue

            # Полная последовательность
            x_i = x[i, :L]
            y_i = y[i, :L]
            full = torch.cat([x_i, y_i[-1:].clone()], dim=0)
            
            full_len = full.size(0)
            
            # 75% на вход, 25% генерируем
            prompt_len = max(1, int(full_len * 0.75))
            target_len = full_len - prompt_len
            
            if target_len < 1:
                continue

            prompt = full[:prompt_len]
            ref_tail = full[prompt_len:]

            gen_full = model.generate(
                input_ids=prompt,
                num_new_tokens=target_len,
                eos_id=None, 
            )

            gen_tail = gen_full[prompt_len:]

            pred_text = tokenizer.decode(gen_tail.tolist(), skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(ref_tail.tolist(), skip_special_tokens=True).strip()

            if pred_text and ref_text:
                predictions.append(pred_text)
                references.append(ref_text)

    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    return rouge.compute(predictions=predictions, references=references)


def main():
    # 1. Загрузка конфига
    cfg = load_yaml("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Подготовка токенизатора и параметров
    model_name = cfg["tokenizer"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    # 3. Инициализация модели
    m_cfg = cfg["model"]
    max_len = int(cfg["train"]["max_len"])
    
    model = LSTMNextToken(
        vocab_size=vocab_size,
        emb_dim=int(m_cfg["emb_dim"]),
        hidden_dim=int(m_cfg["hidden_dim"]),
        num_layers=int(m_cfg["num_layers"]),
        dropout=float(m_cfg["dropout"]),
        pad_id=pad_id,
        max_len=max_len,
    ).to(device)

    # 4. Загрузка весов
    save_path = os.path.join(cfg["save"]["save_dir"], cfg["save"]["save_name"])
    print(f"Loading weights from: {save_path}")
    
    checkpoint = torch.load(save_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 5. Загрузка данных (Валидация из конфига)
    val_csv = cfg["data"]["val_csv"]
    batch_size = int(cfg["train"]["batch_size"])
    
    print(f"Evaluating on: {val_csv}")
    _, val_loader = make_dataloader(
        val_csv, 
        batch_size=batch_size, 
        shuffle=False, 
        max_len=max_len, 
        pad_id=pad_id
    )

    # 6. Расчет Loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    val_loss = eval_loss(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # 7. Расчет ROUGE
    print("Running ROUGE evaluation...")
    rouge_scores = eval_rouge_autocomplete_3_4(
        model=model,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
    )

    print("ROUGE Scores:")
    for k, v in rouge_scores.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()