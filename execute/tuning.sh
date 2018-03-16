python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/sgd_copy_1.csv \
    --save-model    --encoder-path results/enc_sgd_copy_1.pt \
    --decay-times 4 --decoder-path results/dec_sgd_copy_1.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/sgd_copy_2.csv \
    --save-model    --encoder-path results/enc_sgd_copy_2.pt \
    --decay-times 4 --decoder-path results/dec_sgd_copy_2.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/sgd_copy_3.csv \
    --save-model    --encoder-path results/enc_sgd_copy_3.pt \
    --decay-times 4 --decoder-path results/dec_sgd_copy_3.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/sgd_copy_4.csv \
    --save-model    --encoder-path results/enc_sgd_copy_4.pt \
    --decay-times 4 --decoder-path results/dec_sgd_copy_4.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/sgd_copy_4.csv \
    --save-model    --encoder-path results/enc_sgd_copy_4.pt \
    --decay-times 4 --decoder-path results/dec_sgd_copy_4.pt \
    --n-iters 30000  --optimizer SGD

python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_1.csv \
    --save-model    --encoder-path results/enc_sgd_match_1.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_1.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_2.csv \
    --save-model    --encoder-path results/enc_sgd_match_2.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_2.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_3.csv \
    --save-model    --encoder-path results/enc_sgd_match_3.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_3.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_4.csv \
    --save-model    --encoder-path results/enc_sgd_match_4.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_4.pt \
    --n-iters 30000  --optimizer SGD
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/sgd_match_5.csv \
    --save-model    --encoder-path results/enc_sgd_match_5.pt \
    --decay-times 4 --decoder-path results/dec_sgd_match_5.pt \
    --n-iters 30000  --optimizer SGD

python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_1.csv \
    --save-model    --encoder-path results/enc_adam_copy_1.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_1.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_2.csv \
    --save-model    --encoder-path results/enc_adam_copy_2.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_2.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_3.csv \
    --save-model    --encoder-path results/enc_adam_copy_3.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_3.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_4.csv \
    --save-model    --encoder-path results/enc_adam_copy_4.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_4.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type copy \
    --report-results    --results-path results/adam_copy_5.csv \
    --save-model    --encoder-path results/enc_adam_copy_5.pt \
    --decay-times 4 --decoder-path results/dec_adam_copy_5.pt
    --n-iters 30000 --optimizer Adam

python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_1.csv \
    --save-model    --encoder-path results/enc_adam_match_1.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_1.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_2.csv \
    --save-model    --encoder-path results/enc_adam_match_2.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_2.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_3.csv \
    --save-model    --encoder-path results/enc_adam_match_3.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_3.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_4.csv \
    --save-model    --encoder-path results/enc_adam_match_4.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_4.pt
    --n-iters 30000 --optimizer Adam
python 3_advanced.py --task-name challenge --model-type match \
    --report-results    --results-path results/adam_match_5.csv \
    --save-model    --encoder-path results/enc_adam_match_5.pt \
    --decay-times 4 --decoder-path results/dec_adam_match_5.pt
    --n-iters 30000 --optimizer Adam