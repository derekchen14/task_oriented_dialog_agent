python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile1a.csv \
    --n-iters 30000 --decay-times 4 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile1b.csv \
    --n-iters 30000 --decay-times 4 --optimizer Adam

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile2a.csv \
    --weight-decay 0.01 --drop-prob 0.1 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile2b.csv \
    --weight-decay 0.01 --drop-prob 0.1 --optimizer Adam

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile3a.csv \
    --drop-prob 0.2 --optimizer Adam --attn-method dot
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile3b.csv \
    --drop-prob 0.2 --optimizer Adam --attn-method dot

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile4a.csv \
    --drop-prob 0.3 --optimizer Adam --attn-method luong
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile4b.csv \
    --drop-prob 0.3 --optimizer Adam --attn-method luong

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile5a.csv \
    --weight-decay 0.01 --drop-prob 0.2 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/mile5b.csv \
    --weight-decay 0.01 --drop-prob 0.2 --optimizer Adam
