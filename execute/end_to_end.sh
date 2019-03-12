python run.py --task end_to_end --model ddq --max-turn 40 -e 32 \
    --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 14 --warm-start \
    --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
    --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
    --pool-size 5000 --prefix Mar_12_ --suffix _14 --report-qual --verbose

# python run.py --task end_to_end --model ddq --max-turn 40 -e 500 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 15 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix Mar_12_ --suffix _15 --report-qual --verbose

# python run.py --task end_to_end --model ddq --max-turn 40 -e 500 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 16 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix Mar_12_ --suffix _16 --report-qual --verbose
