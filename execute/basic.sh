# python run.py --task woz2 --model basic --model-name fe \
#   --suffix Dec_7 -lr 0.01 --epochs 2 --report-path full_enumeration \
#   --optimizer rmsprop
# python run.py --task woz2 --model basic --model-name po \
#   --suffix Dec_7 -lr 0.01 --epochs 1 --report-path possible_only \
#   --optimizer rmsprop
python run.py --task woz2 --model basic --model-name ov \
  --suffix removeme --epochs 2 --report-path ordered_values

# python dual_run.py --task-name woz2 --model dual --epochs 3 \
#         --suffix Dec_5 --optimizer rmsprop
# python per_slot_run.py --task-name woz2 --model per_slot --epochs 3 \
#         --suffix Dec_4 --optimizer rmsprop

# CUDA_VISIBLE_DEVICES=1 python glad_run.py --gpu=1 --suffix utterance_only3 \
#     --model glad --hidden-size 200 --embedding-size 400 --seed 40 \
#     --learning-rate 1e-3 --epochs 50


# Manual updates: vocab on components and evaluator, preprocess init file