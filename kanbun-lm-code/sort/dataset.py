import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


class KanshiDataset(Dataset):
    def __init__(self, df, model_name_or_path):
        super().__init__()
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.char,
            row.sentence,
            add_special_tokens=True,
            max_length=32,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return (
            ids,
            mask,
            torch.FloatTensor([row.pct_rank]),
        )

    def __len__(self):
        return self.df.shape[0]
