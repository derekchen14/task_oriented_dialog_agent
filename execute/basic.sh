# CUDA_VISIBLE_DEVICES=1 
# python run.py --task dstc2 --model-type dual --model-name dual \
#   --suffix Nov_28 -lr 0.03 --epochs 1 --report-path intent_slot
  # --debug

# python run.py --test-mode --task-name dstc2 --suffix Nov_28 \
#   --report-path intent_slot --model-name is
# python run.py --test-mode --task-name dstc2 --suffix Nov_28 \
#   --report-path per_intent --model-name pi


python per_slot.py --task-name dstc2 --model-type per_slot --epochs 2 \
        --suffix Dec_4 --optimizer rmsprop
# python dual.py --task-name dstc2 --model-type dual --epochs 2 \
#         --suffix Dec_5 --optimizer rmsprop








# Manual updates: change vocab on components and evaluator