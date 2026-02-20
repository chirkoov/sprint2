import os
import yaml
import pandas as pd
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, logging
import torch

logging.set_verbosity_error() # отключение предупреждений при вызове

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_data(csv_path, tokenizer): # загружаем test.csv, смотрим только text, токенизируем для distilgpt2
    df = pd.read_csv(csv_path)
    
    df = df.dropna(subset=["text"])
    texts = df["text"].tolist()
    
    prepared_samples = []
    
    encodings = tokenizer(texts, truncation=True, max_length=256, add_special_tokens=False)
    
    for i, input_ids in enumerate(tqdm(encodings["input_ids"], desc="Filtering")):
        full_len = len(input_ids)
             
        prompt_len = int(full_len * 0.75)
        target_len = full_len - prompt_len

        prompt_ids = input_ids[:prompt_len]
        ref_ids = input_ids[prompt_len:]
        
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
        
        prepared_samples.append({
            "prompt_text": prompt_text,
            "ref_text": ref_text,
            "target_len": target_len # Сколько токенов генерировать
        })
        
    return prepared_samples

def main():
    cfg = load_yaml("configs/config.yaml")
    
    device_id = 0 if torch.cuda.is_available() else -1
    
    test_csv = "data/test.csv"
    data_path = test_csv

    model_name = "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = pipeline(
        "text-generation", 
        model=model_name, 
        tokenizer=tokenizer,
        device=device_id,
    )

    samples = prepare_data(data_path, tokenizer)

    rouge = evaluate.load("rouge")
    predictions = []
    references_text = []
    examples_to_print = []

    for item in tqdm(samples, desc="Processing"):
        prompt = item["prompt_text"]
        ref = item["ref_text"]
        needed_len = item["target_len"]
        
        result = generator(
            prompt, 
            max_new_tokens=needed_len, 
            do_sample=True, 
            top_k=50, 
            pad_token_id=tokenizer.eos_token_id, 
            return_full_text=False 
        )
        
        pred_str = result[0]["generated_text"].strip()
        
        predictions.append(pred_str)
        references_text.append(ref.strip())
        
        if len(examples_to_print) < 3 and len(pred_str) > 5:
            examples_to_print.append((prompt, ref, pred_str))

    rouge_scores = rouge.compute(predictions=predictions, references=references_text)
  
    r1 = rouge_scores.get('rouge1', 0.0)
    r2 = rouge_scores.get('rouge2', 0.0)

    print(
        f"Rouge1: {r1:.4f} | "
        f"Rouge2: {r2:.4f}"
    )  
    
    print("Examples:")
    for i, ex in enumerate(examples_to_print):
        print("")
        print(f"        Example {i+1}")
        print(f"Prompt:    {ex['Prompt']}")
        print(f"Reference: {ex['Reference']}")
        print(f"Generated: {ex['Generated']}")
        print("")

if __name__ == "__main__":
    main()