CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task end_to_end -e 500 \
    --epsilon 0.0 --hidden-dim 100 --embedding-size 200 --use-old-nlu \
    --batch-size 16 --seed 14 --warm-start --model ddq --max-turn 40 \
    --user simulate --metrics success_rate avg_reward --dataset e2e/movies \
    --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
    --pool-size 5000 --prefix their_nlu_ --suffix _14 --verbose

CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task end_to_end -e 500 \
    --epsilon 0.0 --hidden-dim 100 --embedding-size 200 --use-old-nlu \
    --batch-size 16 --seed 15 --warm-start --model ddq --max-turn 40 \
    --user simulate --metrics success_rate avg_reward --dataset e2e/movies \
    --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
    --pool-size 5000 --prefix their_nlu_ --suffix _15 --verbose
    # --early-stop success_rate --use-existing --report-qual

# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task end_to_end -e 600 \
#     --epsilon 0.0 --hidden-dim 100 --embedding-size 200 \
#     --batch-size 16 --seed 15 --warm-start --model ddq --max-turn 40 \
#     --user simulate --metrics success_rate avg_reward --dataset e2e/movies \
#     --learning-rate 1e-3 --optimizer adam --weight-decay 0.0 \
#     --pool-size 5000 --prefix adam_ --suffix _15 --verbose


# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task end_to_end -e 600 \
#     --epsilon 0.0 --hidden-dim 200 --embedding-size 400 \
#     --batch-size 16 --seed 14 --warm-start --model ddq --max-turn 40 \
#     --user simulate --metrics success_rate avg_reward --dataset e2e/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix large_embed_ --suffix _14 --verbose

# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task end_to_end -e 600 \
#     --epsilon 0.0 --hidden-dim 200 --embedding-size 400 \
#     --batch-size 16 --seed 15 --warm-start --model ddq --max-turn 40 \
#     --user simulate --metrics success_rate avg_reward --dataset e2e/movies \
#     --learning-rate 1e-3 --optimizer rmsprop --weight-decay 0.0 \
#     --pool-size 5000 --prefix large_embed_ --suffix _15 --verbose

# CUDA_VISIBLE_DEVICES=0  python run.py --gpu=0 --task track_intent  \
#       --model glad --learning-rate 3e-4 --hidden-dim 200 --embedding-size 400 \
#       --epochs 40 --threshold 0.3 --optimizer adam --dataset e2e/movies \
#       --early-stop joint_goal  --save-model \
#       --prefix nlu_  --suffix _lr1e4 --seed 14 --verbose