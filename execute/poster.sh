python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 \
    --encoder-path results/enc_copy_1a.pt  --encoder-path results/dec_copy_1a.pt \
    --report-results --results-path results/copy_1a.csv -lr 0.01
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 \
    --encoder-path results/enc_copy_1b.pt  --encoder-path results/dec_copy_1b.pt \
    --report-results --results-path results/copy_1b.csv -lr 0.01
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 \
    --encoder-path results/enc_copy_1c.pt  --encoder-path results/dec_copy_1c.pt \
    --report-results --results-path results/copy_1c.csv -lr 0.01

python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 \
    --encoder-path results/enc_copy_2a.pt  --encoder-path results/dec_copy_2a.pt \
    --report-results --results-path results/copy_2a.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 \
    --encoder-path results/enc_copy_2b.pt  --encoder-path results/dec_copy_2b.pt \
    --report-results --results-path results/copy_2b.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 \
    --encoder-path results/enc_copy_2c.pt  --encoder-path results/dec_copy_2c.pt \
    --report-results --results-path results/copy_2c.csv -lr 0.003

python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
    --encoder-path results/enc_copy_3a.pt  --encoder-path results/dec_copy_3a.pt \
    --report-results --results-path results/copy_3a.csv -lr 0.01
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
    --encoder-path results/enc_copy_3b.pt  --encoder-path results/dec_copy_3b.pt \
    --report-results --results-path results/copy_3b.csv -lr 0.01
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
    --encoder-path results/enc_copy_3c.pt  --encoder-path results/dec_copy_3c.pt \
    --report-results --results-path results/copy_3c.csv -lr 0.01

python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
    --encoder-path results/enc_copy_4a.pt  --encoder-path results/dec_copy_4a.pt \
    --report-results --results-path results/copy_4a.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
    --encoder-path results/enc_copy_4b.pt  --encoder-path results/dec_copy_4b.pt \
    --report-results --results-path results/copy_4b.csv -lr 0.003
python 3_advanced.py --task-name challenge --model-type copy \
    --optimizer SGD --save-model --early-stopping 3.2 --weight-decay 0.01 \
    --encoder-path results/enc_copy_4c.pt  --encoder-path results/dec_copy_4c.pt \
    --report-results --results-path results/copy_4c.csv -lr 0.003