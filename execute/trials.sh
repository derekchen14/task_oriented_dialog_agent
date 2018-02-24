python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy4a.csv --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy4b.csv --optimizer Adam

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy5a.csv \
    --weight-decay 0.0 --drop-prob 0.1
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy5b.csv \
    --weight-decay 0.0 --drop-prob 0.1

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy6a.csv \
    --n-iters 30000 --decay-times 4 --attn-method vinyals
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy6b.csv \
    --n-iters 30000 --decay-times 4 --attn-method vinyals

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy7a.csv \
    --n-iters 30000 --decay-times 4 --attn-method luong
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy7b.csv \
    --n-iters 30000 --decay-times 4 --attn-method luong

