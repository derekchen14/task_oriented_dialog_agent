python run.py --task policy --model rulebased --report-quant -e 5 \
    --user simulate --metrics success_rate avg_reward --dataset ddq/movies \
    --batch-size 16 --verbose  # --debug
# python run.py --task policy --model rulebased -e 5 --user command \
#     --dataset ddq/movies --batch-size 16 --verbose


# python run.py --task policy --model rulebased --report-quant -e 5 --user simulate \
#     --metrics success_rate avg_reward --dataset ddq/movies --batch-size 17

# python run.py --task policy --model rulebased --report-quant -e 5 --user simulate \
#     --metrics success_rate avg_reward --dataset ddq/movies --batch-size 18