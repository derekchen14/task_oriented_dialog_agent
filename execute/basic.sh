# CUDA_VISIBLE_DEVICES=1 
# python run.py --task dstc2 --model-type basic \
#   --suffix Nov_28 -lr 0.03 --epochs 2 --report-path full_enumeration \
#   --report-results --save-model

python run.py --test-mode --task-name dstc2 --suffix Nov_28 \
  --report-path full_enumeration --model-name fe
# python run.py --test-mode --task-name dstc2 --suffix Nov_28 \
#   --report-path ordered_values --model-name ov
# python run.py --test-mode --task-name dstc2 --suffix Nov_28 \
#   --report-path possible_only --model-name po
