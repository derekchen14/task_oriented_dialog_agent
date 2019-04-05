python run.py --task manage_policy --model ddq --max-turn 40 -e 500 \
    --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 14 --warm-start \
    --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
    --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
    --pool-size 5000 --prefix Apr_05_unittest_ --suffix _14 --verbose
    # --early-stop success_rate --use-existing --report-qual

# python run.py --task manage_policy --model ddq --max-turn 40 -e 500 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 15 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix Mar_15_unittest_ --suffix _15 --report-qual --verbose

# python run.py --task manage_policy --model ddq --max-turn 40 -e 500 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 16 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix Mar_15_unittest_ --suffix _16 --report-qual --verbose


# python run.py --task policy --model rulebased --report-quant -e 5 \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --batch-size 16 --verbose  --debug