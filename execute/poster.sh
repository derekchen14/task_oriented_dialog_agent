CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer Adam --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_1a.pt  --encoder-path results/dec_vanilla_1a.pt \
    --report-results --results-path results/vanilla_1a.csv -lr 0.01
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer Adam --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_1b.pt  --encoder-path results/dec_vanilla_1b.pt \
    --report-results --results-path results/vanilla_1b.csv -lr 0.01
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer Adam --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_1c.pt  --encoder-path results/dec_vanilla_1c.pt \
    --report-results --results-path results/vanilla_1c.csv -lr 0.01

CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer Adam --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_2a.pt  --encoder-path results/dec_vanilla_2a.pt \
    --report-results --results-path results/vanilla_2a.csv -lr 0.003
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer Adam --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_2b.pt  --encoder-path results/dec_vanilla_2b.pt \
    --report-results --results-path results/vanilla_2b.csv -lr 0.003
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer Adam --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_2c.pt  --encoder-path results/dec_vanilla_2c.pt \
    --report-results --results-path results/vanilla_2c.csv -lr 0.003

CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer SGD --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_3a.pt  --encoder-path results/dec_vanilla_3a.pt \
    --report-results --results-path results/vanilla_3a.csv -lr 0.01
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer SGD --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_3b.pt  --encoder-path results/dec_vanilla_3b.pt \
    --report-results --results-path results/vanilla_3b.csv -lr 0.01
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer SGD --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_3c.pt  --encoder-path results/dec_vanilla_3c.pt \
    --report-results --results-path results/vanilla_3c.csv -lr 0.01


CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer SGD --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_4a.pt  --encoder-path results/dec_vanilla_4a.pt \
    --report-results --results-path results/vanilla_4a.csv -lr 0.003
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer SGD --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_4b.pt  --encoder-path results/dec_vanilla_4b.pt \
    --report-results --results-path results/vanilla_4b.csv -lr 0.003
CUDA_VISIBLE_DEVICES=3 python 3_advanced.py --task-name challenge --model-type gru \
    --optimizer SGD --save-model --early-stopping 2.8 \
    --encoder-path results/enc_vanilla_4c.pt  --encoder-path results/dec_vanilla_4c.pt \
    --report-results --results-path results/vanilla_4c.csv -lr 0.003