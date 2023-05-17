source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.8/3.8.13 cuda/11.3/11.3.1

VENV_DIR=pathto/venv
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip3 install --upgrade pip
pip3 install torch==1.12.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install transformers==4.21.0 sklearn tqdm pandas numpy
pip3 install protobuf==3.20.0 sentencepiece datasets accelerate
pip3 install zenhan fugashi unidic unidic-lite mecab-python3 bs4 nltk sumeval janome bert-score
pip3 list

mkdir -p outputs
cd outputs
mkdir -p tmp_model
mkdir -p sorted
cd sorted
mkdir -p label
mkdir -p prediction
cd ../../

MODEL_LIST=(
    "google/mt5-small"
    "google/mt5-base"
    "google/mt5-large"
)

BATCH_SIZE=(8 16 32)
LEARNING_RATE=(1e-5 2e-5 5e-5)
EPOCH=(10 20 30)

for model in ${MODEL_LIST[@]}; do
    for bs in ${BATCH_SIZE[@]}; do
        for lr in ${LEARNING_RATE[@]}; do
            for epoch in ${EPOCH[@]}; do
                python3 train.py \
                    --train_path pathto/dataset/train.csv \
                    --val_path pathto/dataset/val.csv \
                    --test_path pathto/dataset/test.csv \
                    --model_name_or_path $model \
                    --batch_size $bs \
                    --learning_rate $lr \
                    --accumulation_steps 2 \
                    --epochs $epoch \
                    --sort_hakubun \
                    --sort_with label
            done
        done
    done
done
