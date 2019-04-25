python run.py --task end_to_end --model ddq --max-turn 40 -e 40 \
    --epsilon 0.0 --hidden-dim 100 --embedding-size 200 \
    --batch-size 16 --seed 14 --warm-start \
    --user simulate --metrics success_rate avg_reward --dataset e2e/movies \
    --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
    --pool-size 5000 --prefix draft_ --suffix _apr18 --verbose
    # --early-stop success_rate --use-existing --report-qual

# python run.py --task end_to_end --model ddq --max-turn 40 -e 500 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 15 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix Mar_15_unittest_ --suffix _15 --report-qual --verbose

# python run.py --task end_to_end --model ddq --max-turn 40 -e 500 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 16 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix Mar_15_unittest_ --suffix _16 --report-qual --verbose

# CUDA_VISIBLE_DEVICES=0  python run.py --gpu=0 --task track_intent  \
#       --model glad --learning-rate 3e-4 --hidden-dim 200 --embedding-size 400 \
#       --epochs 40 --threshold 0.3 --optimizer adam --dataset e2e/movies \
#       --early-stop joint_goal  --save-model \
#       --prefix nlu_  --suffix _lr1e4 --seed 14 --verbose