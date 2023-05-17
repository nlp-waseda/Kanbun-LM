source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.8/3.8.13 cuda/11.3/11.3.1

VENV_DIR=pathto/venv
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip3 install --upgrade pip
pip3 install torch==1.12.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install transformers==4.21.0 sklearn tqdm pandas numpy
pip3 install protobuf==3.20.0 sentencepiece datasets accelerate
pip3 install zenhan fugashi unidic unidic-lite mecab-python3 bs4 nltk
pip3 list

MODEL_LIST=(
    "cl-tohoku/bert-base-japanese-char-v2"
    "ku-nlp/roberta-base-japanese-char-wwm"
    "bert-base-chinese"
    "hfl/chinese-bert-wwm-ext"
    "KoichiYasuoka/roberta-classical-chinese-base-char"
    "KoichiYasuoka/roberta-classical-chinese-large-char"
)

BATCH_SIZE=(8 16 32)
LEARNING_RATE=(1e-5 2e-5 5e-5)
EPOCH=20

for model in ${MODEL_LIST[@]}; do
    for bs in ${BATCH_SIZE[@]}; do
        for lr in ${LEARNING_RATE[@]}; do
            python3 train.py \
                --train_path pathto/dataset/train.csv \
                --val_path pathto/dataset/val.csv \
                --test_path pathto/dataset/test.csv \
                --model_name_or_path $model \
                --batch_size $bs \
                --learning_rate $lr \
                --accumulation_steps 2 \
                --epochs $EPOCH
        done
    done
done
