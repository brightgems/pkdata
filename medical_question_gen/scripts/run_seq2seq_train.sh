export PYTHONPATH=$(pwd)

python pytorch-seq2seq-attention.py \
    --train \
    --epochs 100 \
    --learning_rate 2e-3 \
    --early_stopping 5 \
    --batch_size 1 \
    --save_every 2\
    --dropout_p 0.5
