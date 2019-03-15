python run.py --task end_to_end --model ddq --max-turn 40 --seed 15 \
    --hidden-dim 80 --metrics success_rate avg_reward \
    --user command --early-stop success_rate --dataset ddq/movies \
    --prefix saver_ --suffix _15 --report-qual --verbose --test-mode


