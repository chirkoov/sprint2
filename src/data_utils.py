import re
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


# функция для "чистки" текстов
def clean_string(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = text.lower() # приведение к нижнему регистру
    text = re.sub(r'[^a-z0-9\s]', '', text) # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'\s+', ' ', text).strip() # удаление дублирующихся пробелов, удаление пробелов по краям
    return text


def preprocess_dataset(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    df = pd.DataFrame({"text": lines})
    df["text"] = df["text"].astype(str).apply(clean_string)
    df.to_csv(output_path, index=False)
    
    
def tokenize_dataset(input_path: str, output_path: str):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(input_path)

    texts = df["text"].fillna("").astype(str).tolist()

    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True
    )


    # сохраняем input_ids как строку чисел через пробел
    df["input_ids"] = [
        " ".join(map(str, ids)) for ids in tokenized["input_ids"]
    ]

    df.to_csv(output_path, index=False)


def split_dataset(input_path: str,
                  train_path: str,
                  val_path: str,
                  test_path: str,
                  random_state: int = 42):

    df = pd.read_csv(input_path)

    # 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        shuffle=True
    )

    # 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        shuffle=True
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)