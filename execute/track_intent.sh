# Unit test for original GLAD
# CUDA_VISIBLE_DEVICES=0  python run.py --gpu=0 --task track_intent  \
#       --model glad --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 \
#       --epochs 50 --threshold 0.3 --optimizer adam --dataset woz2 \
#       --early-stop joint_goal  --save-model \
#       --prefix apr30_  --suffix _unittest_14 --seed 14 --verbose

# CUDA_VISIBLE_DEVICES=0  python run.py --gpu=0 --task track_intent  \
#       --model glad --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 \
#       --epochs 50 --threshold 0.3 --optimizer adam --dataset woz2 \
#       --early-stop joint_goal  --save-model \
#       --prefix apr30_  --suffix _unittest_15 --seed 14 --verbose

# Hyperparameter Tuning
# CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent  \
#       --model glad --learning-rate 3e-4 --hidden-dim 200 --embedding-size 400 \
#       --epochs 40 --threshold 0.3 --optimizer adam --dataset e2e/movies \
#       --early-stop joint_goal  --save-model \
#       --prefix nlu_  --suffix _lr3e3 --seed 14 --verbose
# CUDA_VISIBLE_DEVICES=0  python run.py --gpu=0 --task track_intent --dataset woz2 \
#       --model glad --learning-rate 3e-4 --hidden-dim 200 --embedding-size 400 \
#       --epochs 50 --threshold 0.3 --optimizer adam --early-stop success_rate \
#       --prefix Apr_08_  --suffix _1 --seed 14    # --debug
# CUDA_VISIBLE_DEVICES=0  python run.py --gpu=0 --task track_intent  \
#       --model glad --learning-rate 3e-4 --hidden-dim 200 --embedding-size 400 \
#       --epochs 1 --threshold 0.3 --optimizer adam --dataset e2e/movies \
#       --early-stop joint_goal  --report-qual --test-mode \
#       --prefix nlu_  --suffix _apr9 --seed 14  --verbose  # --use-existing

# Evaluation
python run.py --gpu=0 --task track_intent \
      --model glad --learning-rate 3e-4 --hidden-dim 200 --embedding-size 400 \
      --epochs 1 --threshold 0.3 --dataset e2e/movies --use-existing \
      --early-stop joint_goal  --report-qual --test-mode \
      --prefix old_nlu_  --suffix _14 --seed 14  --verbose
