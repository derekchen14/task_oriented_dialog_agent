python run.py --task end_to_end --model ddq --max-turn 40 --seed 35 \
    --hidden-dim 80 --metrics success_rate avg_reward \
    --user command --early-stop success_rate --dataset ddq/movies \
    --prefix saver_ --suffix _15 --report-qual --verbose --test-mode

# python run.py --task end_to_end --model ddq --max-turn 40 -e 7 \
#     --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 15 \
#     --user simulate --early-stop success_rate --dataset ddq/movies \
#     --prefix saver_ --suffix _15 --report-qual --verbose --debug --test-mode
