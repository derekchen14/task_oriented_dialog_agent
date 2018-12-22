# python run.py --task full_enumeration --dataset basicwoz --model basic \
#   --prefix removeme_ -lr 0.001 --epochs 2 --optimizer rmsprop
# python run.py --task full_enumeration --dataset basicwoz --model basic \
#   --prefix removeme_ --suffix _pretrained -lr 0.001 --epochs 2 \
#   --optimizer rmsprop --pretrained
# python run.py --task ordered_values --dataset basicwoz --model basic \
#   --prefix removeme_ --learning-rate 0.001 --epochs 2 --optimizer rmsprop
# python run.py --task possible_only --dataset basicwoz --model basic \
#   --prefix removeme_ --suffix adam -lr 0.001 --epochs 2 --optimizer adam
# python run.py --task ordered_values --dataset basicwoz --model bilstm \
#   --prefix removeme_ -lr 0.001 --epochs 2 --optimizer sgd

python run.py --task ordered_values --dataset basicwoz --model bilstm \
  --suffix _best_model -lr 0.001 --epochs 1 --optimizer adam --use-existing
python run.py --task ordered_values --dataset basicwoz --model bilstm \
  --suffix _best_model -lr 0.001 --optimizer rmsprop --test-mode
python run.py --task full_enumeration --dataset basicwoz --model basic \
  --prefix removeme_ -lr 0.001 --epochs 2 --optimizer rmsprop --debug
