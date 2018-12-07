python run.py --task woz2 --model-type basic --model-name fe \
  --suffix Dec_7 -lr 0.01 --epochs 2 --report-path full_enumeration \
  --optimizer rmsprop
python run.py --task woz2 --model-type basic --model-name po \
  --suffix Dec_7 -lr 0.01 --epochs 2 --report-path possible_only \
  --optimizer rmsprop
python run.py --task woz2 --model-type basic --model-name ov \
  --suffix Dec_7 -lr 0.01 --epochs 2 --report-path ordered_values \
  --optimizer rmsprop

# python dual_run.py --task-name woz2 --model-type dual --epochs 1 \
#         --suffix Dec_5 --optimizer rmsprop
# python per_slot_run.py --task-name woz2 --model-type per_slot --epochs 1 \
#         --suffix Dec_4 --optimizer rmsprop

# CUDA_VISIBLE_DEVICES=1 python glad_run.py --epoch 10


# Manual updates: vocab on components and evaluator, preprocess init file