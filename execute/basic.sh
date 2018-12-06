# python run.py --task dstc2 --model-type basic --model-name fe \
#   --suffix Dec_5 -lr 0.03 --epochs 2 --report-path full_enumeration \
#   --optimizer rmsprop
# python run.py --task dstc2 --model-type basic --model-name po \
#   --suffix Dec_5 -lr 0.03 --epochs 2 --report-path possible_only \
#   --optimizer rmsprop
# python run.py --task dstc2 --model-type basic --model-name ov \
#   --suffix Dec_5 -lr 0.01 --epochs 2 --report-path ordered_values \
#   --optimizer rmsprop

CUDA_VISIBLE_DEVICES=1 python glad_run.py --epoch 10

# python per_slot.py --task-name dstc2 --model-type per_slot --epochs 2 \
#         --suffix Dec_4 --optimizer rmsprop
# python dual.py --task-name dstc2 --model-type dual --epochs 2 \
#         --suffix Dec_5 --optimizer rmsprop



# Manual updates: change vocab on components and evaluator