python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_1a.pt  --encoder-path results/dec_attn_1a.pt \
    --report-results --results-path results/attn_1a.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_1b.pt  --encoder-path results/dec_attn_1b.pt \
    --report-results --results-path results/attn_1b.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_1c.pt  --encoder-path results/dec_attn_1c.pt \
    --report-results --results-path results/attn_1c.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_1d.pt  --encoder-path results/dec_attn_1d.pt \
    --report-results --results-path results/attn_1d.csv --n-iters 30000

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_2a.pt  --encoder-path results/dec_attn_2a.pt \
    --report-results --results-path results/attn_2a.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_2b.pt  --encoder-path results/dec_attn_2b.pt \
    --report-results --results-path results/attn_2b.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_2c.pt  --encoder-path results/dec_attn_2c.pt \
    --report-results --results-path results/attn_2c.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_2d.pt  --encoder-path results/dec_attn_2d.pt \
    --report-results --results-path results/attn_2d.csv --n-iters 30000

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_3a.pt  --encoder-path results/dec_attn_3a.pt \
    --report-results --results-path results/attn_3a.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_3b.pt  --encoder-path results/dec_attn_3b.pt \
    --report-results --results-path results/attn_3b.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_3c.pt  --encoder-path results/dec_attn_3c.pt \
    --report-results --results-path results/attn_3c.csv --n-iters 30000
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_3d.pt  --encoder-path results/dec_attn_3d.pt \
    --report-results --results-path results/attn_3d.csv --n-iters 30000

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_4a.pt  --encoder-path results/dec_attn_4a.pt \
    --report-results --results-path results/attn_4a.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_4b.pt  --encoder-path results/dec_attn_4b.pt \
    --report-results --results-path results/attn_4b.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_4c.pt  --encoder-path results/dec_attn_4c.pt \
    --report-results --results-path results/attn_4c.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_4d.pt  --encoder-path results/dec_attn_4d.pt \
    --report-results --results-path results/attn_4d.csv -lr 0.003

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_5a.pt  --encoder-path results/dec_attn_5a.pt \
    --report-results --results-path results/attn_5a.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_5b.pt  --encoder-path results/dec_attn_5b.pt \
    --report-results --results-path results/attn_5b.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_5c.pt  --encoder-path results/dec_attn_5c.pt \
    --report-results --results-path results/attn_5c.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_5d.pt  --encoder-path results/dec_attn_5d.pt \
    --report-results --results-path results/attn_5d.csv -lr 0.003

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_6a.pt  --encoder-path results/dec_attn_6a.pt \
    --report-results --results-path results/attn_6a.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_6b.pt  --encoder-path results/dec_attn_6b.pt \
    --report-results --results-path results/attn_6b.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_6c.pt  --encoder-path results/dec_attn_6c.pt \
    --report-results --results-path results/attn_6c.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_6d.pt  --encoder-path results/dec_attn_6d.pt \
    --report-results --results-path results/attn_6d.csv -lr 0.003

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_7a.pt  --encoder-path results/dec_attn_7a.pt \
    --report-results --results-path results/attn_7a.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_7b.pt  --encoder-path results/dec_attn_7b.pt \
    --report-results --results-path results/attn_7b.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_7c.pt  --encoder-path results/dec_attn_7c.pt \
    --report-results --results-path results/attn_7c.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method luong \
    --encoder-path results/enc_attn_7d.pt  --encoder-path results/dec_attn_7d.pt \
    --report-results --results-path results/attn_7d.csv --n-layers 1

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_8a.pt  --encoder-path results/dec_attn_8a.pt \
    --report-results --results-path results/attn_8a.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_8b.pt  --encoder-path results/dec_attn_8b.pt \
    --report-results --results-path results/attn_8b.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_8c.pt  --encoder-path results/dec_attn_8c.pt \
    --report-results --results-path results/attn_8c.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method dot \
    --encoder-path results/enc_attn_8d.pt  --encoder-path results/dec_attn_8d.pt \
    --report-results --results-path results/attn_8d.csv --n-layers 1

python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_9a.pt  --encoder-path results/dec_attn_9a.pt \
    --report-results --results-path results/attn_9a.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_9b.pt  --encoder-path results/dec_attn_9b.pt \
    --report-results --results-path results/attn_9b.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_9c.pt  --encoder-path results/dec_attn_9c.pt \
    --report-results --results-path results/attn_9c.csv --n-layers 1
python 3_advanced.py --task-name challenge --model-type attention \
    --optimizer SGD --save-model --early-stopping 2.1 --attn-method vinyals \
    --encoder-path results/enc_attn_9d.pt  --encoder-path results/dec_attn_9d.pt \
    --report-results --results-path results/attn_9d.csv --n-layers 1