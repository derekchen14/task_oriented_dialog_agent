python 3_advanced.py --task-name challenge --model-type transformer \
    --report-results  --results-path results/testing_1.csv \
    --decay-times 3 --n-layers 6 --n-iters 30000 \
    --optimizer Adam --early-stopping 2.8
python 3_advanced.py --task-name challenge --model-type transformer \
    --report-results  --results-path results/testing_2.csv \
    --decay-times 3 --n-layers 6 --n-iters 30000 \
    --optimizer Adam --early-stopping 3.5
python 3_advanced.py --task-name challenge --model-type transformer \
    --report-results  --results-path results/testing_3.csv \
    --decay-times 3 --n-layers 6 --n-iters 30000 \
    --optimizer Adam --early-stopping 3.5

python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_1.csv \
    --save-model    --encoder-path results/enc_sgd_match_1.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_1.pt \
    --n-iters 30000  --optimizer SGD --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_2.csv \
    --save-model    --encoder-path results/enc_sgd_match_2.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_2.pt \
    --n-iters 30000  --optimizer SGD --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_3.csv \
    --save-model    --encoder-path results/enc_sgd_match_3.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_3.pt \
    --n-iters 30000  --optimizer SGD --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_4.csv \
    --save-model    --encoder-path results/enc_sgd_match_4.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_4.pt \
    --n-iters 30000  --optimizer SGD --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_5.csv \
    --save-model    --encoder-path results/enc_sgd_match_5.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_5.pt \
    --n-iters 30000  --optimizer SGD --early-stopping 1.8

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_1.csv \
    --save-model    --encoder-path results/enc_adam_copy_1.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_1.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_2.csv \
    --save-model    --encoder-path results/enc_adam_copy_2.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_2.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_3.csv \
    --save-model    --encoder-path results/enc_adam_copy_3.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_3.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_4.csv \
    --save-model    --encoder-path results/enc_adam_copy_4.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_4.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_5.csv \
    --save-model    --encoder-path results/enc_adam_copy_5.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_5.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8

python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_1.csv \
    --save-model    --encoder-path results/enc_adam_match_1.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_1.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_2.csv \
    --save-model    --encoder-path results/enc_adam_match_2.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_2.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_3.csv \
    --save-model    --encoder-path results/enc_adam_match_3.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_3.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_4.csv \
    --save-model    --encoder-path results/enc_adam_match_4.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_4.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_5.csv \
    --save-model    --encoder-path results/enc_adam_match_5.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_5.pt
    --n-iters 30000 --optimizer Adam --early-stopping 1.8
