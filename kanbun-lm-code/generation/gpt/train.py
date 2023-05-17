import argparse
import json

import pandas as pd
import torch
from bert_score import score
from datasets import Dataset
from nltk import bleu_score, ribes_score, word_tokenize
from sumeval.metrics.rouge import RougeCalculator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

parser = argparse.ArgumentParser(description="Process some arguments")
parser.add_argument("--model_name_or_path", type=str, default="sberbank-ai/mGPT")
parser.add_argument("--train_path", type=str, default="../../dataset/train.csv")
parser.add_argument("--val_path", type=str, default="../../dataset/val.csv")
parser.add_argument("--test_path", type=str, default="../../dataset/test.csv")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--accumulation_steps", type=int, default=2)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--sort_hakubun", action="store_true")
parser.add_argument(
    "--sort_with",
    default="label",
    choices=["label", "prediction"],
)
parser.add_argument(
    "--sort_with_prediction_data",
    type=str,
    default=None,
)

args = parser.parse_args()


def idx2int(idx):
    if "1" <= idx and idx <= "9":
        return ord(idx) - ord("0")
    else:
        return ord(idx) - ord("A") + 1


def sort_hakubun(hakubun, order):
    order = str(order)
    sorted_hakubun = ""
    for idx in order:
        sorted_hakubun += hakubun[idx2int(idx) - 1]
    return sorted_hakubun


def _preprocess_data(args, tokenizer):
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)

    if args.sort_hakubun:
        train_df["hakubun"] = train_df.apply(
            lambda x: sort_hakubun(x["hakubun"], x["reading_order_ja"]), axis=1
        )
        val_df["hakubun"] = val_df.apply(
            lambda x: sort_hakubun(x["hakubun"], x["reading_order_ja"]), axis=1
        )

        if args.sort_with == "label":
            test_df["hakubun"] = test_df.apply(
                lambda x: sort_hakubun(x["hakubun"], x["reading_order_ja"]), axis=1
            )
        else:
            assert args.sort_with_prediction_data is not None

            def read_json(file_path):
                with open(file_path) as f:
                    return json.load(f)

            json_data = read_json(args.sort_with_prediction_data)
            test_pred_order = []
            for data in json_data:
                test_pred_order.append(data["pred_order"])
            test_df["reading_order_ja"] = test_pred_order
            test_df["hakubun"] = test_df.apply(
                lambda x: sort_hakubun(x["hakubun"], x["reading_order_ja"]), axis=1
            )

    train_df["data"] = train_df["hakubun"] + "$" + train_df["kakikudashi"]
    val_df["data"] = val_df["hakubun"] + "$" + val_df["kakikudashi"]

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_dataset_tokenized = train_dataset.map(
        lambda examples: tokenizer(
            examples["data"], truncation=True, padding="max_length", max_length=32
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset_tokenized = val_dataset.map(
        lambda examples: tokenizer(
            examples["data"], truncation=True, padding="max_length", max_length=32
        ),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    return train_dataset_tokenized, val_dataset_tokenized, test_df


def predict(tokenizer, test_df, output_dir=None):
    model = AutoModelForCausalLM.from_pretrained("./outputs/tmp_model")

    predictions = []
    for text in tqdm(test_df["hakubun"].tolist()):
        text_tokenized = tokenizer(
            text + "$",
            return_tensors="pt",
        ).input_ids
        output_ids = model.generate(
            text_tokenized,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            bad_word_ids=[[tokenizer.unk_token_id]],
        )
        prediction = tokenizer.decode(output_ids.tolist()[0])
        predictions.append(prediction.strip().split("$")[1])

    labels = test_df["kakikudashi"].tolist()

    bleu_scores = []
    ribes_scores = []
    rouge_l_scores = []

    rouge = RougeCalculator(lang="ja")
    for (pred, label) in zip(predictions, labels):
        ref = [word_tokenize(" ".join(label))]
        hyp = word_tokenize(" ".join(pred))
        bleu_scores.append(bleu_score.sentence_bleu(ref, hyp))
        ribes_scores.append(ribes_score.sentence_ribes(ref, hyp))
        rouge_l_scores.append(
            rouge.rouge_l(summary=" ".join(pred), references=" ".join(label))
        )

    bleu = sum(bleu_scores) / len(bleu_scores)
    ribes = sum(ribes_scores) / len(ribes_scores)
    P, R, F1 = score(
        predictions,
        labels,
        lang="ja",
        verbose=True,
        model_type="cl-tohoku/bert-base-japanese-char-v2",
        num_layers=11,
    )
    bertscore = F1.mean().item()
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    print(f"BLEU score is: {bleu}")
    print(f"RIBES score is: {ribes}")
    print(f"BERTscore is: {bertscore}")
    print(f"ROUGE-L score is: {rouge_l}")

    if output_dir is not None:
        prefix = "outputs/"
        if args.sort_hakubun:
            prefix += "sorted/"
            if args.sort_with == "label":
                prefix += "label/"
            else:
                prefix += "prediction/"

        output_dir = (
            prefix
            + f"bleu_{round(bleu, 4)}_ribes_{round(ribes, 4)}_bertsocre_{round(bertscore, 4)}_rougel_{round(rouge_l, 4)}_{output_dir}.json"
        )
        with open(
            output_dir,
            "w",
        ) as f:
            json_list = []
            for (haku, label, pred) in zip(test_df["hakubun"], labels, predictions):
                json_list.append({"haku": haku, "label": label, "pred": pred})
            json.dump(json_list, f, ensure_ascii=False)


def train(
    args,
    tokenizer,
    model,
    train_dataset_tokenized,
    val_dataset_tokenized,
    test_df,
):
    training_args = TrainingArguments(
        output_dir="./outputs/tmp_model",
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_strategy="epoch",
        save_strategy="no",
    )

    class DataCollatorForComet(DataCollatorForLanguageModeling):
        def torch_call(self, examples):
            batch = super().torch_call(examples)
            labels = batch["labels"]

            new_labels = []

            for lbls in labels:
                labels_int = lbls.tolist()

                for i in range(labels_int.index(10) + 1):  # "$" is NO.10
                    labels_int[i] = -100

                new_labels.append(torch.LongTensor(labels_int))

            batch["labels"] = torch.stack(new_labels)
            return batch

    data_collator = DataCollatorForComet(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    del trainer

    output_dir = f"MODEL_{args.model_name_or_path.replace('/', '_')}_BS_{args.batch_size}_LR_{args.learning_rate}_EPOCH_{args.epochs}"

    predict(tokenizer, test_df, output_dir=output_dir)


def main():
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    (
        train_dataset_tokenized,
        val_dataset_tokenized,
        test_df,
    ) = _preprocess_data(args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    train(
        args,
        tokenizer,
        model,
        train_dataset_tokenized,
        val_dataset_tokenized,
        test_df,
    )


if __name__ == "__main__":
    main()
