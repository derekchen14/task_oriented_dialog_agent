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
