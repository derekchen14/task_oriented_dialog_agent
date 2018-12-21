python run.py --task full_enumeration --dataset basicwoz --model basic \
  --prefix removeme_ -lr 0.01 --epochs 2 --optimizer rmsprop
python run.py --task full_enumeration --dataset basicwoz --model bilstm \
  --prefix removeme_ -lr 0.01 --epochs 2 --optimizer rmsprop
# python run.py --task ordered_values --dataset basicwoz --model basic \
#   --prefix removeme_ --suffix rmsprop -lr 0.01 --epochs 2 --optimizer rmsprop
# python run.py --task possible_only --dataset basicwoz --model basic \
#   --prefix removeme_ --suffix adam -lr 0.01 --epochs 2 --optimizer adam