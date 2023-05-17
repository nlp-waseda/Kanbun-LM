import torch.nn as nn
from transformers import AutoModel


class KanshiModel(nn.Module):
    def __init__(self, model_path):
        super(KanshiModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = self.top(x[:, 0, :])
        return x
