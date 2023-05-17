import re

import deplacy
import nltk
import pandas as pd
import udkanbun
import udkanbun.kaeriten
import udkundoku
from bert_score import score
from metrics import kendall_tau
from nltk import bleu_score, ribes_score, word_tokenize
from sumeval.metrics.rouge import RougeCalculator
from tqdm import tqdm


def validate(val_df):
    lzh = udkundoku.load(Danku=False)

    labels = []
    predictions = []

    bleu_scores = []
    ribes_scores = []
    rouge_l_scores = []

    rouge = RougeCalculator(lang="ja")

    for idx, row in tqdm(val_df.iterrows()):
        sentence = row["hakubun"]
        s = lzh(sentence)
        label = row["kakikudashi"]

        try:
            ud_trans = str(udkundoku.translate(s))
            pred = re.search(r"# text = (.*?)\n", ud_trans).group(1)

            labels.append(label)
            predictions.append(pred)

            ref = [word_tokenize(" ".join(label))]
            hyp = word_tokenize(" ".join(pred))
            bleu_scores.append(bleu_score.sentence_bleu(ref, hyp))
            ribes_scores.append(ribes_score.sentence_ribes(ref, hyp))
            rouge_l_scores.append(
                rouge.rouge_l(summary=" ".join(pred), references=" ".join(label))
            )
        except:
            pass

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
    bertscore = F1.mean().float()
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    print(f"BLEU score is: {bleu}")
    print(f"RIBES score is: {ribes}")
    print(f"BERTscore is: {bertscore}")
    print(f"ROUGE-L score is: {rouge_l}")

    print(f"Totally {len(val_df) - len(bleu_scores)} weren't translated successfully.")


def main():
    val_df = pd.read_csv("pathto/dataset/test.csv")
    validate(val_df)


if __name__ == "__main__":
    main()
