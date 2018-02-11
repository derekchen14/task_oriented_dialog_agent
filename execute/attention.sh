# python 3_seq2seq.py --task-name challenge --hidden-size 256 --n-layers 1 \
#     --drop-prob 0.0 -enp 'results/seq_en.pt' -edp 'results/seq_de.pt' \
#     -ep 'results/seq_error.csv' -s -teach 0.0

python 4_attention.py -task-name challenge --attn_method luong \
    --report-results --results-path results/luong1.csv
python 4_attention.py -task-name challenge --drop-prob 0.25 \
    --report-results --results-path results/luong2.csv
python 4_attention.py -task-name challenge --drop-prob 0.3 \
    --report-results --results-path results/luong3.csv

python 4_attention.py -task-name challenge --attn_method vinyals \
    --report-results --results-path results/vinyals1.csv
python 4_attention.py -task-name challenge --drop-prob 0.25 \
    --report-results --results-path results/vinyals2.csv
python 4_attention.py -task-name challenge --drop-prob 0.3 \
    --report-results --results-path results/vinyals3.csv

python 4_attention.py -task-name challenge --attn_method dot \
    --report-results --results-path results/dot1.csv
python 4_attention.py -task-name challenge --drop-prob 0.25 \
    --report-results --results-path results/dot2.csv
python 4_attention.py -task-name challenge --drop-prob 0.3 \
    --report-results --results-path results/dot3.csv

