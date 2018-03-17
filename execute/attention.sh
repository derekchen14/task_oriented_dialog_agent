python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_poster1a.pt  --encoder-path results/dec_poster1a.pt \
    --report-results --results-path results/poster1a.csv -lr 0.0158
python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_poster1b.pt  --encoder-path results/dec_poster1b.pt \
    --report-results --results-path results/poster1b.csv -lr 0.0158

python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_poster2a.pt  --encoder-path results/dec_poster2a.pt \
    --report-results --results-path results/poster2a.csv -lr 0.001
python 3_advanced.py --task-name challenge --model-type transformer \
    --optimizer Adam --n-layers 6 --n-iters 30000 --save-model \
    --encoder-path results/enc_poster2b.pt  --encoder-path results/dec_poster2b.pt \
    --report-results --results-path results/poster2b.csv -lr 0.001