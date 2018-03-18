python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_1 --early-stopping 1.8 \
    --report-results --results-path combined_1 --n-iters 30000 \
    --optimizer SGD --attn-method luong --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_2 --early-stopping 1.8 \
    --report-results --results-path combined_2 --n-iters 30000 \
    --optimizer SGD --attn-method vinyals --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_3 --early-stopping 1.8 \
    --report-results --results-path combined_3 --n-iters 30000 \
    --optimizer SGD --attn-method dot --weight-decay 0.003 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_4 --early-stopping 1.8 \
    --report-results --results-path combined_4 --n-iters 30000 \
    --optimizer SGD --attn-method luong --weight-decay 0.01 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_5 --early-stopping 1.8 \
    --report-results --results-path combined_5 --n-iters 30000 \
    --optimizer SGD --attn-method vinyals --weight-decay 0.01 --decay-times 3
python 3_advanced.py --task-name challenge --model-type combined \
    --save-model --model-path combined_6 --early-stopping 1.8 \
    --report-results --results-path combined_6 --n-iters 30000 \
    --optimizer SGD --attn-method dot --weight-decay 0.01 --decay-times 3
