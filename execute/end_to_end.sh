python run.py --task end_to_end --model ddq --max-turn 40 -e 500 \
    --epsilon 0.0 --hidden-dim 80 --batch-size 16 --seed 14 --warm-start \
    --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
    --pool-size 5000 --prefix Mar_2_ --suffix _remove # --verbose