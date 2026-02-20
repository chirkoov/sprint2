import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def _parse_input_ids(s: str) -> list[int]: # input_ids в csv сохранены как строка чисел через пробел
    if pd.isna(s):
        return []
    return [int(x) for x in str(s).strip().split() if x]


class NextTokenDataset(Dataset):
    def __init__(self, csv_path: str, max_len: int = 256):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        ids = _parse_input_ids(self.df.loc[idx, "input_ids"])

        ids = ids[: self.max_len]
        if len(ids) < 2:
            ids = ids + [0]  

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        return {"input_ids": x, "targets": y}


def collate_fn(batch, pad_id: int = 0):
    xs = [item["input_ids"] for item in batch]
    ys = [item["targets"] for item in batch]

    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)

    x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=pad_id)

    return {
        "input_ids": x_padded,
        "targets": y_padded,
        "lengths": lengths,
    }


def make_dataloader(csv_path: str, batch_size: int, shuffle: bool, max_len: int = 256, pad_id: int = 0):
    dataset = NextTokenDataset(csv_path=csv_path, max_len=max_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )
    return dataset, loader
