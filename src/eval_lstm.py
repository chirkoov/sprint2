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
def test_val_loss(model, loader, criterion, device):
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
def eval_rouge_autocomplete_3_4(model, loader, tokenizer, device, max_examples=3):
    model.eval()
    rouge = evaluate.load("rouge")

    predictions = []
    references = []
    examples = []

    for batch in tqdm(loader, desc="Evaluating Rouge", leave=True):
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        lengths = batch["lengths"].to(device)

        B = x.size(0)
        for i in range(B):
            L = int(lengths[i].item())
            if L <= 2:
                continue

            # вся последовательность
            x_i = x[i, :L]
            y_i = y[i, :L]
            full = torch.cat([x_i, y_i[-1:].clone()], dim=0)
            
            full_len = full.size(0)
            
            # 75% на вход, 25% генерируем
            prompt_len = max(1, int(full_len * 0.75))
            target_len = full_len - prompt_len
            
            prompt = full[:prompt_len]
            ref_tail = full[prompt_len:]

            gen_full = model.generate(
                input_ids=prompt,
                num_new_tokens=target_len,
                eos_id=None, 
            )

            gen_tail = gen_full[prompt_len:]
            
            prompt_text = tokenizer.decode(prompt.tolist(), skip_special_tokens=True).strip()
            pred_text = tokenizer.decode(gen_tail.tolist(), skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(ref_tail.tolist(), skip_special_tokens=True).strip()

            if pred_text and ref_text:
                predictions.append(pred_text)
                references.append(ref_text)

                if len(examples) < max_examples:
                    examples.append({
                        "Prompt": prompt_text,
                        "Reference": ref_text,
                        "Generated": pred_text,
                    })

    scores = rouge.compute(predictions=predictions, references=references)
    return scores, examples


def main():
    cfg = load_yaml("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["tokenizer"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

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

    save_path = os.path.join(cfg["save"]["save_dir"], cfg["save"]["save_name"])
    
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    test_csv = cfg["data"]["test_csv"]
    batch_size = int(cfg["train"]["batch_size"])
    
    _, test_loader = make_dataloader(
        test_csv, 
        batch_size=batch_size, 
        shuffle=False, 
        max_len=max_len, 
        pad_id=pad_id
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    test_loss = test_val_loss(model, test_loader, criterion, device)

    rouge_scores, examples_to_print = eval_rouge_autocomplete_3_4(
    model=model,
    loader=test_loader,
    tokenizer=tokenizer,
    device=device,
)
    r1 = rouge_scores.get('rouge1', 0.0)
    r2 = rouge_scores.get('rouge2', 0.0)

    print(
        f"Val Loss: {test_loss:.4f} | "
        f"Rouge1: {r1:.4f} | "
        f"Rouge2: {r2:.4f}"
    )  
    
    print("Examples:")
    for i, ex in enumerate(examples_to_print):
        print(f"        Example {i+1}")
        print(f"Prompt:    {ex['Prompt']}")
        print(f"Reference: {ex['Reference']}")
        print(f"Generated: {ex['Generated']}")
        print("")

if __name__ == "__main__":
    main()