python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.1 --attn-method luong --decay-times 2 \
    --report-results --results-path results/tune_1a.csv
python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.1 --attn-method luong --decay-times 2 \
    --report-results --results-path results/tune_1b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.2 --attn-method luong --weight-decay 0.01 \
    --report-results --results-path results/tune_2a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.2 --attn-method luong --weight-decay 0.01 \
    --report-results --results-path results/tune_2b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.2 --attn-method vinyals --weight-decay 0.01 \
    --report-results --results-path results/tune_3a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.2 --attn-method vinyals --weight-decay 0.01 \
    --report-results --results-path results/tune_3b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.2 --attn-method dot --weight-decay 0.01 \
    --report-results --results-path results/tune_3a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.2 --attn-method dot --weight-decay 0.01 \
    --report-results --results-path results/tune_3b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method luong --learning-rate 0.003 \
    --report-results --results-path results/tune_4a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method luong --learning-rate 0.003 \
    --report-results --results-path results/tune_4b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method vinyals --learning-rate 0.003 \
    --report-results --results-path results/tune_5a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method vinyals --learning-rate 0.03 \
    --report-results --results-path results/tune_5b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method luong --decay-times 3 \
    --report-results --results-path results/tune_6a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method luong --decay-times 3 \
    --report-results --results-path results/tune_6b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method vinyals --decay-times 3 \
    --report-results --results-path results/tune_7a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method vinyals --decay-times 3 \
    --report-results --results-path results/tune_7b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method luong --decay-times 2 \
    --report-results --results-path results/tune_8a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method luong --decay-times 2 \
    --report-results --results-path results/tune_8b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method vinyals --decay-times 2 \
    --report-results --results-path results/tune_9a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method vinyals --decay-times 2 \
    --report-results --results-path results/tune_9b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method dot --decay-times 2 \
    --report-results --results-path results/tune_10a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.1 --attn-method dot --decay-times 2 \
    --report-results --results-path results/tune_10b.csv