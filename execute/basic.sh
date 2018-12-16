# python run.py --task woz2 --model-type basic --model-name fe \
#   --suffix Dec_7 -lr 0.01 --epochs 2 --report-path full_enumeration \
#   --optimizer rmsprop
# python run.py --task woz2 --model-type basic --model-name po \
#   --suffix Dec_7 -lr 0.01 --epochs 1 --report-path possible_only \
#   --optimizer rmsprop
python run.py --task woz2 --model-type basic --model-name ov \
  --suffix embeddings1 -lr 0.0003 --epochs 2 --report-path ordered_values \
  --optimizer adam --drop-prob 0.2 --seed 14 --embedding-size 400
python run.py --task woz2 --model-type basic --model-name ov \
  --suffix embeddings2 -lr 0.0003 --epochs 2 --report-path ordered_values \
  --optimizer adam --drop-prob 0.2 --seed 15 --embedding-size 400

# python dual_run.py --task-name woz2 --model-type dual --epochs 3 \
#         --suffix Dec_5 --optimizer rmsprop
# python per_slot_run.py --task-name woz2 --model-type per_slot --epochs 3 \
#         --suffix Dec_4 --optimizer rmsprop

# CUDA_VISIBLE_DEVICES=1 python glad_run.py --gpu=1 --suffix utterance_only3 \
#     --model-type glad --hidden-size 200 --embedding-size 400 --seed 40 \
#     --learning-rate 1e-3 --epochs 50


# Manual updates: vocab on components and evaluator, preprocess init file
