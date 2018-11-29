# CUDA_VISIBLE_DEVICES=1 
python run.py --task dstc2 --model-type dual --model-name dual \
  --suffix Nov_28 -lr 0.03 --epochs 1 --report-path intent_slot
  # --debug

# python run.py --test-mode --task-name dstc2 --suffix Nov_28 \
#   --report-path intent_slot --model-name is
# python run.py --test-mode --task-name dstc2 --suffix Nov_28 \
#   --report-path per_intent --model-name pi