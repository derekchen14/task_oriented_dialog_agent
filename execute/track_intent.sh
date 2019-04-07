CUDA_VISIBLE_DEVICES=1  python run.py --task track_intent --dataset woz2 --model glad \
      --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 --epochs 50 --gpu=1 \
      --threshold 0.3 --seed 14 --prefix Apr_05_  --suffix _1 --optimizer adam
      # --debug  # --stop-early accuracy
#     --gpu=1  --prefix temp --hidden-size 200 --embedding-size 400 \
#     --learning-rate 1e-3 --optimizer adam
# CUDA_VISIBLE_DEVICES=1 python run.py --task track_intent --dataset woz2 \
#      --model glad --gpu=1 \
#      --learning-rate 1e-3 --hidden-dim 200 --embedding-size 400 --epochs 30 \
#      --threshold 0.3 --seed 15 --prefix Apr_04_ --suffix _4 --optimizer adam
