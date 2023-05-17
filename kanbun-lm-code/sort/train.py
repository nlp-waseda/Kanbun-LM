import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import KanshiDataset
from metrics import kendall_tau
from model import KanshiModel

parser = argparse.ArgumentParser(description="Process some arguments")
parser.add_argument(
    "--model_name_or_path", type=str, default="cl-tohoku/bert-base-japanese-v2"
)
parser.add_argument("--train_path", type=str, default="../dataset/train.csv")
parser.add_argument("--val_path", type=str, default="../dataset/val.csv")
parser.add_argument("--test_path", type=str, default="../dataset/test.csv")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--accumulation_steps", type=int, default=2)
parser.add_argument("--epochs", type=int, default=2)

args = parser.parse_args()

if not os.path.exists("./outputs"):
    os.mkdir("./outputs")


def _preprocess_data(df):

    data_dict = {
        "sentence_id": [],
        "char_id": [],
        "char": [],
        "sentence": [],
        "pct_rank": [],
        "real_order": [],
    }

    for idx1, row in df.iterrows():
        haku = row["hakubun"]
        ordr = str(row["reading_order_ja"])
        for idx2, ch in enumerate(haku):
            char_id = str(idx2 + 1)
            if len(char_id) > 1:
                char_id = chr(ord("A") + (idx2 + 1 - 10))

            rank = ordr.index(char_id)
            pct_rank = rank / (len(haku) - 1)

            data_dict["sentence_id"].append(idx1 + 1)
            data_dict["char_id"].append(char_id)
            data_dict["char"].append(ch + char_id)
            data_dict["sentence"].append(haku)
            data_dict["pct_rank"].append(pct_rank)
            data_dict["real_order"].append(ordr)

    new_df = pd.DataFrame(data=data_dict)

    return new_df


def load_data(args):
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)

    train_df = _preprocess_data(train_df)
    val_df = _preprocess_data(val_df)
    test_df = _preprocess_data(test_df)

    train_ds = KanshiDataset(
        train_df,
        model_name_or_path=args.model_name_or_path,
    )
    val_ds = KanshiDataset(
        val_df,
        model_name_or_path=args.model_name_or_path,
    )
    test_ds = KanshiDataset(
        test_df,
        model_name_or_path=args.model_name_or_path,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, drop_last=True, collate_fn=default_convert
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, drop_last=False, collate_fn=default_convert
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, drop_last=False, collate_fn=default_convert
    )

    return train_df, val_df, test_df, train_loader, val_loader, test_loader


def _set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def _read_data(data):
    ids = []
    mask = []
    target = []

    for d in data:
        ids.append(torch.LongTensor(d[0]))
        mask.append(torch.LongTensor(d[1]))
        target.append(d[2])

    ids = torch.stack(ids).cuda()
    mask = torch.stack(mask).cuda()
    target = torch.stack(target).cuda()

    return (ids, mask), target


def validate(model, val_df, val_loader, output_dir=None):
    model.eval()
    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = _read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())

    real_order = (
        val_df.groupby("sentence_id")["real_order"].max().astype(str).apply(list)
    )
    val_df["pred"] = np.concatenate(preds)
    val_df["char_id"] = val_df["char_id"].astype(str)
    y_dummy = val_df.sort_values("pred").groupby("sentence_id")["char_id"].apply(list)
    score = kendall_tau(real_order, y_dummy)
    print("Preds score", score)

    if output_dir is not None:
        with open(
            f"outputs/normal-input/{round(score, 4)}_{output_dir}.json", "w"
        ) as f:
            json_list = []
            for idx, sub_df in val_df.groupby("sentence_id"):
                json_list.append(
                    {
                        "sentence_id": str(sub_df["sentence_id"].iloc[0]),
                        "sentence": sub_df["sentence"].iloc[0],
                        "real_order": sub_df["real_order"].iloc[0],
                        "pred_order": "".join(y_dummy[idx]),
                    }
                )
            json.dump(json_list, f, ensure_ascii=False)

    return score


def train(args, model, val_df, test_df, train_loader, val_loader, test_loader):
    _set_seed()
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_optimization_steps = int(
        args.epochs * len(train_loader) / args.accumulation_steps
    )
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
    )  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    best_score = 0
    best_epoch = None
    best_model = None

    for e in range(args.epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = _read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            if idx % 100 == 0:
                print(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

        score = validate(model, val_df, val_loader)
        if best_score < score:
            best_score = score
            best_epoch = e
            best_model = copy.deepcopy(model)

    output_dir = f"MODEL_{args.model_name_or_path.replace('/', '_')}_BS_{args.batch_size}_LR_{args.learning_rate}_EPOCH_{best_epoch}"

    validate(best_model, test_df, test_loader, output_dir=output_dir)


def main(args):
    train_df, val_df, test_df, train_loader, val_loader, test_loader = load_data(args)

    model = KanshiModel(args.model_name_or_path)
    model = model.cuda()

    train(args, model, val_df, test_df, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main(args)
