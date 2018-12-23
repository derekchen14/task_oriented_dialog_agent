CUDA_VISIBLE_DEVICES=2 python run.py --task dual --dataset basicwoz \
    --model basic --epochs 2 --gpu=2
CUDA_VISIBLE_DEVICES=2 python run.py --task per_slot --dataset basicwoz \
    --model basic --epochs 2 --gpu=2

# python run.py --task glad --dataset woz2 --model glad --learning-rate 1e-3 \
#       --hidden-size 200 --embedding-size 400 --epochs 2
CUDA_VISIBLE_DEVICES=1 python run.py --task glad --dataset woz2 --model glad \
    --gpu=1  --prefix temp --hidden-size 200 --embedding-size 400 \
    --learning-rate 1e-3
CUDA_VISIBLE_DEVICES=2 python run.py --task glad --dataset woz2 --model glad \
    --gpu=2  --prefix temp --hidden-size 200 --embedding-size 400 --test-mode


python run.py --task full_enumeration --dataset basicwoz --model basic \
  --prefix removeme_ -lr 0.001 --epochs 2 --optimizer rmsprop --gpu=2
python run.py --task full_enumeration --dataset basicwoz --model basic \
  --prefix removeme_ --suffix _pretrained -lr 0.001 --epochs 2 \
  --optimizer rmsprop --pretrained --gpu=2
python run.py --task ordered_values --dataset basicwoz --model basic  --gpu=2 \
  --prefix removeme_ --learning-rate 0.001 --epochs 2 --optimizer rmsprop
python run.py --task possible_only --dataset basicwoz --model basic  --gpu=2 \
  --prefix removeme_ --suffix adam -lr 0.001 --epochs 2 --optimizer adam
python run.py --task ordered_values --dataset basicwoz --model bilstm \
  --prefix removeme_ -lr 0.001 --epochs 2 --optimizer sgd  --gpu=2
