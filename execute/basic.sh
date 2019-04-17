# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent --dataset woz2 \
#       --model glad --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 \
#       --epochs 50 --threshold 0.3 --optimizer adam --save-model \
#       --prefix Apr_09_  --suffix _1 --seed 14  --verbose  # --debug

# python run.py --task full_enumeration --dataset basicwoz --model basic \
#   --prefix removeme_ -lr 0.001 --epochs 2 --optimizer rmsprop


# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent  \
#       --model glad --learning-rate 3e-3 --hidden-dim 200 --embedding-size 400 \
#       --epochs 40 --threshold 0.3 --optimizer adam --dataset e2e/movies \
#       --early-stop joint_goal  --save-model \
#       --prefix nlu_  --suffix _lr3e3 --seed 14 --verbose

# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent  \
#       --model glad --learning-rate 1e-2 --hidden-dim 200 --embedding-size 400 \
#       --epochs 40 --threshold 0.3 --optimizer adam --dataset e2e/movies \
#       --early-stop joint_goal  --save-model \
#       --prefix nlu_  --suffix _lr1e2 --seed 14 --verbose


# Pre-existing model for evaluation, this save_dir has a working CPU version
python run.py --task track_intent --early-stop joint_goal  --test-mode \
      --model glad --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 \
      --epochs 1 --threshold 0.3 --optimizer adam --dataset woz2 \
      --prefix Apr_08_  --suffix _14 --seed 14  --verbose --use-existing
