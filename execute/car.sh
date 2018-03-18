# python 3_advanced.py --task-name challenge --model-type replica \
#     --save-model --model-path replica --early-stopping 1.8 \
#     --report-results --results-path replica --n-iters 30000 \
#     --optimizer Adam --attn-method vinyals --hidden-size 352

# python 3_advanced.py -t 1 --debug  --model-type replica \
#     --optimizer Adam --attn-method vinyals --hidden-size 352

python 3_advanced.py --task-name challenge --model-type gru --test-mode
python 3_advanced.py --task-name challenge --model-type attention --test-mode
python 3_advanced.py --task-name challenge --model-type copy --test-mode
