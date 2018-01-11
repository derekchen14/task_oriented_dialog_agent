# python 3_seq2seq.py --task-name challenge --hidden-size 256 --n-layers 1 \
#     --drop-prob 0.0 -enp 'results/seq_en.pt' -edp 'results/seq_de.pt' \
#     -ep 'results/seq_error.csv' -s -teach 0.0

python 4_attention.py --task-name challenge --hidden-size 256 \
    --n-layers 1 --drop-prob 0.3 --teacher-forcing 0.6 --debug -v \
    --encoder-path 'results/attn_enc.pt' --decoder-path 'results/attn_dec.pt' \
    --save-errors --error-path 'results/attn_error.csv'


