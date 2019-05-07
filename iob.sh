python train_old_nlu.py --task track_intent --epochs 20 --dataset e2e/movies \
     --prefix iob_ --model lstm --suffix _14 --seed 14 --learning-rate 0.00003 \
     --save-model
python train_old_nlu.py --task track_intent --epochs 20 --dataset e2e/movies \
     --prefix iob_ --model lstm --suffix _15 --seed 15 --learning-rate 0.00003 \
     --save-model
python train_old_nlu.py --task track_intent --epochs 20 --dataset e2e/movies \
     --prefix iob_ --model lstm --suffix _16 --seed 16 --learning-rate 0.00003 \
     --save-model

python evaluate_old_nlu.py --task track_intent --dataset e2e/movies \
     --prefix iob_ --model lstm --suffix _14 --seed 14
python evaluate_old_nlu.py --task track_intent --dataset e2e/movies \
     --prefix iob_ --model lstm --suffix _15 --seed 15
python evaluate_old_nlu.py --task track_intent --dataset e2e/movies \
     --prefix iob_ --model lstm --suffix _16 --seed 16
