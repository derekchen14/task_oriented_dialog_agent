# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent --dataset woz2 \
#       --model glad --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 \
#       --epochs 50 --threshold 0.3 --optimizer adam --save-model \
#       --prefix Apr_09_  --suffix _1 --seed 14  --verbose  # --debug
# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent  \
#       --model glad --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 \
#       --epochs 1 --threshold 0.3 --optimizer adam --dataset woz2 \
#       --early-stop joint_goal  --report-qual --test-mode \
#       --prefix Apr_09_  --suffix _15 --seed 15  --verbose  # --use-existing

# python run.py --task full_enumeration --dataset basicwoz --model basic \
#   --prefix removeme_ -lr 0.001 --epochs 2 --optimizer rmsprop


CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent  \
      --model glad --learning-rate 3e-3 --hidden-dim 200 --embedding-size 400 \
      --epochs 40 --threshold 0.3 --optimizer adam --dataset e2e/movies \
      --early-stop joint_goal  --save-model \
      --prefix nlu_  --suffix _lr3e3 --seed 14 --verbose

CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent  \
      --model glad --learning-rate 1e-2 --hidden-dim 200 --embedding-size 400 \
      --epochs 40 --threshold 0.3 --optimizer adam --dataset e2e/movies \
      --early-stop joint_goal  --save-model \
      --prefix nlu_  --suffix _lr1e2 --seed 14 --verbose