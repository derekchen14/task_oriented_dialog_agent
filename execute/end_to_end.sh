python run.py --task end_to_end --model ddq --max-turn 40 --seed 15 \
    --hidden-dim 80 --metrics success_rate avg_reward \
    --user command --early-stop success_rate --dataset ddq/movies \
    --prefix saver_ --suffix _15 --report-qual --verbose --test-mode

# python run.py --task end_to_end --model ddq --max-turn 40 -e 400 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 14 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix fruit_ --suffix _14 --report-qual --verbose \
#     --early-stop success_rate --use-existing

# python run.py --task end_to_end --model ddq --max-turn 40 -e 400 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 15 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix fruit_ --suffix _15 --report-qual --verbose

# python run.py --task end_to_end --model ddq --max-turn 40 -e 400 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 16 --warm-start \
#     --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix fruit_ --suffix _16 --report-qual --verbose
