python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_transformer_1a.pt  --decoder-path results/dec_transformer_1a.pt \
    --report-results --results-path results/transformer_1a.csv -lr 0.0158
python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_transformer_1b.pt  --decoder-path results/dec_transformer_1b.pt \
    --report-results --results-path results/transformer_1b.csv -lr 0.0158

python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_transformer_2a.pt  --decoder-path results/dec_transformer_2a.pt \
    --report-results --results-path results/transformer_2a.csv -lr 0.001
python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_transformer_2b.pt  --decoder-path results/dec_transformer_2b.pt \
    --report-results --results-path results/transformer_2b.csv -lr 0.001

# python 3_advanced.py --task-name challenge --model-type copy \
#     --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
#     --encoder-path results/enc_copy_4a.pt  --encoder-path results/dec_copy_4a.pt \
#     --report-results --results-path results/copy_4a.csv -lr 0.003
# python 3_advanced.py --task-name challenge --model-type copy \
#     --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
#     --encoder-path results/enc_copy_4b.pt  --encoder-path results/dec_copy_4b.pt \
#     --report-results --results-path results/copy_4b.csv -lr 0.003
# python 3_advanced.py --task-name challenge --model-type copy \
#     --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
#     --encoder-path results/enc_copy_4c.pt  --encoder-path results/dec_copy_4c.pt \
#     --report-results --results-path results/copy_4c.csv -lr 0.003
