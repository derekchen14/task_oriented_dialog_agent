python run.py --task end_to_end --model ddq --max-turn 40 -e 10 \
    --pool-size 5000 --prefix saver_ --suffix _15 --report-qual --verbose \
    --epsilon 0.0 --hidden-dim 80 --seed 16 --dataset ddq/movies \
    --early-stop success_rate --test-mode  --user turk