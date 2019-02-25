# python run.py --task glad --dataset woz2 --model glad --verbose \
#       --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 --epochs 50 \
#       --threshold 0.3 --seed 14 --prefix Feb_24_  --suffix _3 --optimizer adam
      # CUDA_VISIBLE_DEVICES=1  --gpu 1
      # --debug  # --stop-early accuracy
#     --gpu=1  --prefix temp --hidden-size 200 --embedding-size 400 \
#     --learning-rate 1e-3 --optimizer adam
CUDA_VISIBLE_DEVICES=1 python run.py --task glad --dataset woz2 --model glad --gpu=1 \
      --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 --epochs 30 \
      --threshold 0.3 --seed 15 --prefix Feb_24_ --suffix _4 --optimizer adam
