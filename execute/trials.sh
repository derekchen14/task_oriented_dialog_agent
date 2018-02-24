python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy1a.csv \
    --weight-decay 0.0 --drop-prob 0.0
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy1b.csv \
    --weight-decay 0.0 --drop-prob 0.0

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy2a.csv
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy2b.csv

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy3a.csv \
    --teacher-forcing 0.8
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy3b.csv \
    --teacher-forcing 0.8

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy4a.csv --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results --results-path results/copy4b.csv --optimizer Adam
