python run.py --task policy --model rulebased --report-quant \
    --metrics success_rate reward --dataset ddq/movies --batch-size 16

# python run.py --task-name simulation --model-type rule --report-results \
#     --metrics success_rate --dataset ddq/restaurants

# python run.py --task-name simulation --model-type rule --report-results \
#     --metrics success_rate --dataset ddq/taxis
    # avg_turn reward


# python run.py --task-name simulation --model-type attention --report-results \
#     --optimizer rmsprop --debug
# python 3_advanced.py -t 1 --debug  --model-type replica \
#     --optimizer Adam --attn-method vinyals --hidden-size 352
