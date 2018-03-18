python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_01 --early-stopping 1.8 \
    --report-results --results-path combined_01 --n-iters 30000 \
    --optimizer SGD --attn-method luong --weight-decay 0.01 --decay-times 2
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_02 --early-stopping 1.8 \
    --report-results --results-path combined_02 --n-iters 30000 \
    --optimizer Adam --attn-method luong --weight-decay 0.01 --decay-times 2
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_03 --early-stopping 1.8 \
    --report-results --results-path combined_03 --n-iters 30000 \
    --optimizer SGD --attn-method vinyals --weight-decay 0.01 --decay-times 2
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_04 --early-stopping 1.8 \
    --report-results --results-path combined_04 --n-iters 30000 \
    --optimizer Adam --attn-method vinyals --weight-decay 0.01 --decay-times 2

python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_05 --early-stopping 1.8 \
    --report-results --results-path combined_05 --n-iters 30000 \
    --optimizer SGD --attn-method luong --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_06 --early-stopping 1.8 \
    --report-results --results-path combined_06 --n-iters 30000 \
    --optimizer Adam --attn-method luong --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_07 --early-stopping 1.8 \
    --report-results --results-path combined_07 --n-iters 30000 \
    --optimizer SGD --attn-method vinyals --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_08 --early-stopping 1.8 \
    --report-results --results-path combined_08 --n-iters 30000 \
    --optimizer Adam --attn-method vinyals --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_09 --early-stopping 1.8 \
    --report-results --results-path combined_09 --n-iters 30000 \
    --optimizer SGD --attn-method dot --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_10 --early-stopping 1.8 \
    --report-results --results-path combined_10 --n-iters 30000 \
    --optimizer Adam --attn-method dot --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_11 --early-stopping 1.8 \
    --report-results --results-path combined_11 --n-iters 30000 \
    --optimizer SGD --attn-method luong --weight-decay 0.01 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_12 --early-stopping 1.8 \
    --report-results --results-path combined_12 --n-iters 30000 \
    --optimizer Adam --attn-method luong --weight-decay 0.01 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_13 --early-stopping 1.8 \
    --report-results --results-path combined_13 --n-iters 30000 \
    --optimizer SGD --attn-method vinyals --weight-decay 0.01 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_14 --early-stopping 1.8 \
    --report-results --results-path combined_14 --n-iters 30000 \
    --optimizer Adam --attn-method vinyals --weight-decay 0.01 --decay-times 3
