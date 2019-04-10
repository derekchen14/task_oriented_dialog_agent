CUDA_VISIBLE_DEVICES=1  python run.py --gpu=1 --task track_intent --dataset woz2 \
      --model glad --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 \
      --epochs 50 --threshold 0.3 --optimizer adam --early-stop success_rate \
      --prefix Apr_09_  --suffix _1 --seed 14    # --debug

# python run.py --task full_enumeration --dataset basicwoz --model basic \
#   --prefix removeme_ -lr 0.001 --epochs 2 --optimizer rmsprop
