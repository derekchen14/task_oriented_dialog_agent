python 4_attention.py --task-name challenge --attn-method luong \
    --report-results --results-path results/luong1.csv
python 4_attention.py --task-name challenge --attn-method luong \
    --report-results --results-path results/luong2.csv
python 4_attention.py --task-name challenge --attn-method luong \
    --report-results --results-path results/luong3.csv

python 4_attention.py --task-name challenge --attn-method vinyals \
    --report-results --results-path results/vinyals1.csv
python 4_attention.py --task-name challenge --attn-method vinyals \
    --report-results --results-path results/vinyals2.csv
python 4_attention.py --task-name challenge --attn-method vinyals \
    --report-results --results-path results/vinyals3.csv

python 4_attention.py --task-name challenge --attn-method dot \
    --report-results --results-path results/dot1.csv
python 4_attention.py --task-name challenge --attn-method dot \
    --report-results --results-path results/dot2.csv
python 4_attention.py --task-name challenge --attn-method dot \
    --report-results --results-path results/dot3.csv

