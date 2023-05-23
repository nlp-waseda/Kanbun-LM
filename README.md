# Kanbun-LM

This is the repository of our paper "Kanbun-LM: Reading and Translating Classical Chinese in Japanese Methods by Language Models". Our paper was accepted by Findings of ACL 2023, see you in Toronto!

[[arXiv]](https://arxiv.org/abs/2305.12759) [[GitHub]](https://github.com/nlp-waseda/Kanbun-LM) [demo]

## Dataset

- We introduce this dataset mainly in Section 3 "Our Dataset and Tasks".

- There are three files for train, validation, and test. We split the dataset using group shuffle split to ensure that all sentences in one poem would not be split.
- Each file contains 4 columns:
  - `poetry_id`: The ids of poem, each poem has multiple sentences.
  - `hakubun`: The original Classical Chinese sentences.
  - `kakikudashi`: The translated Kanbun sentences.
  - `reading_order_ja`: The Japanese reading orders of the original sentences (the numbers represent their index in the original text).

## Code

- We introduce our implementation mainly in Section 4.1 "Implementation for Tasks".

- There are three folders.
  - `baseline` is the implementation for baseline `UD-Kundoku`. Please check the original repository for more details: https://github.com/KoichiYasuoka/UD-Kundoku.
  - `sort` is the implementation for the character reordering task.
    - Grid search details could be found in `sort/run.sh`.
  - `generation` is the implementation for the machine translation task.
    - T5 and GPT do not share codes, please check `generation/t5` and `generation/gpt` separately for more details.
    - Grid search details could be found in `generation/t5/run.sh` and `generation/gpt/run.sh`.
    - The pipeline was implemented by `--sort_hakubun` option. Use `--sort_with label` to do pre-reorder by gold labels, use `--sort_with prediction` to do pre-reorder by prediction results.

## Citation

```tex
@misc{wang2023kanbunlm,
      title={Kanbun-LM: Reading and Translating Classical Chinese in Japanese Methods by Language Models}, 
      author={Hao Wang and Hirofumi Shimizu and Daisuke Kawahara},
      year={2023},
      eprint={2305.12759},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

