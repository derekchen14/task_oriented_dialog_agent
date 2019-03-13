python run.py --task end_to_end --model ddq --max-turn 40 -e 500 \
    --pool-size 5000 --prefix saver_ --suffix _15 --report-qual --verbose \
    --epsilon 0.0 --hidden-dim 80 --seed 15 --dataset ddq/movies \
    --metrics success_rate avg_reward --test-mode  --user command