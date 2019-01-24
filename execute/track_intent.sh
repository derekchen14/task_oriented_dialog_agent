CUDA_VISIBLE_DEVICES=2 python run.py --task glad --dataset woz2 --model glad \
      --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 --epochs 50 \
      --threshold 0.3 --seed 14 --prefix Jan_24_  --suffix _5 --gpu 2 --optimizer adam
      # --debug
#     --gpu=1  --prefix temp --hidden-size 200 --embedding-size 400 \
#     --learning-rate 1e-3 --optimizer adam
# CUDA_VISIBLE_DEVICES=2 python run.py --task glad --dataset woz2 --model glad \
#     --gpu=2  --prefix temp --hidden-size 200 --embedding-size 400 --test-mode
