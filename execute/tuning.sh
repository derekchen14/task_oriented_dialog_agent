python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.4 --attn-method luong --decay-times 2 \
    --report-results --results-path results/drop4_attn1a.csv
python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.4 --attn-method luong --decay-times 2 \
    --report-results --results-path results/drop4_attn1b.csv

python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.3 --attn-method luong --decay-times 2 \
    --report-results --results-path results/drop3_attn2a.csv
python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.3 --attn-method luong --decay-times 2 \
    --report-results --results-path results/drop3_attn2b.csv

python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.4 --attn-method vinyals --decay-times 2 \
    --report-results --results-path results/drop4_v_attn3a.csv
python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.4 --attn-method vinyals --decay-times 2 \
    --report-results --results-path results/drop4_v_attn3b.csv

python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.3 --attn-method vinyals --decay-times 2 \
    --report-results --results-path results/drop3_v_attn4a.csv
python 4_attention.py --task-name challenge --model-type attention \
    --drop-prob 0.3 --attn-method vinyals --decay-times 2 \
    --report-results --results-path results/drop3_v_attn4b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method luong --weight-decay 0.01 \
    --report-results --results-path results/drop4_decay1_1a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method luong --weight-decay 0.01 \
    --report-results --results-path results/drop4_decay1_1b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method luong --weight-decay 0.01 \
    --report-results --results-path results/drop3_decay1_2a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method luong --weight-decay 0.01 \
    --report-results --results-path results/drop3_decay1_2b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method vinyals --weight-decay 0.01 \
    --report-results --results-path results/drop4_decay1_3a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method vinyals --weight-decay 0.01 \
    --report-results --results-path results/drop4_decay1_3b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method vinyals --weight-decay 0.01 \
    --report-results --results-path results/drop3_decay1_4a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method vinyals --weight-decay 0.01 \
    --report-results --results-path results/drop3_decay1_4b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method luong --weight-decay 0.03 \
    --report-results --results-path results/drop4_decay3_5a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method luong --weight-decay 0.03 \
    --report-results --results-path results/drop4_decay3_5b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method luong --weight-decay 0.03 \
    --report-results --results-path results/drop3_decay3_6a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method luong --weight-decay 0.03 \
    --report-results --results-path results/drop3_decay3_6b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method vinyals --weight-decay 0.03 \
    --report-results --results-path results/drop4_decay3_7a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.4 --attn-method vinyals --weight-decay 0.03 \
    --report-results --results-path results/drop4_decay3_7b.csv

python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method vinyals --weight-decay 0.03 \
    --report-results --results-path results/drop3_decay3_8a.csv
python 4_attention.py --task-name challenge --model-type match \
    --drop-prob 0.3 --attn-method vinyals --weight-decay 0.03 \
    --report-results --results-path results/drop3_decay3_8b.csv