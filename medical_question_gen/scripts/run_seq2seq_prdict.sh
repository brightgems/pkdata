export PYTHONPATH=$(pwd)

python pytorch-seq2seq-attention.py \
    --predict \
    --checkpoint_dir=checkpoints/seq2seq_attn/2020-09-22_1728 \
    --batch_size 64
