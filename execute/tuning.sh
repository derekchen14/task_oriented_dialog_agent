python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_1a.csv \
    --encoder-path results/enc_1a.pt --decoder-path results/dec_1a.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_1b.csv \
    --encoder-path results/enc_1b.pt --decoder-path results/dec_1b.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_1c.csv \
    --encoder-path results/enc_1c.pt --decoder-path results/dec_1c.pt

python 4_attention.py --task-name challenge --decay-times 3 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method luong --report-results \
    --model-type match --results-path results/tune_2a.csv \
    --encoder-path results/enc_2a.pt --decoder-path results/dec_2a.pt
python 4_attention.py --task-name challenge --decay-times 3 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method luong --report-results \
    --model-type match --results-path results/tune_2b.csv \
    --encoder-path results/enc_2b.pt --decoder-path results/dec_2b.pt
python 4_attention.py --task-name challenge --decay-times 3 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method luong --report-results \
    --model-type match --results-path results/tune_2c.csv \
    --encoder-path results/enc_2c.pt --decoder-path results/dec_2c.pt

python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type attention --results-path results/tune_3a.csv \
    --encoder-path results/enc_3a.pt --decoder-path results/dec_3a.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type attention --results-path results/tune_3b.csv \
    --encoder-path results/enc_3b.pt --decoder-path results/dec_3b.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type attention --results-path results/tune_3c.csv \
    --encoder-path results/enc_3c.pt --decoder-path results/dec_3c.pt

python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method dot --report-results \
    --model-type match --results-path results/tune_4a.csv \
    --encoder-path results/enc_4a.pt --decoder-path results/dec_4a.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method dot --report-results \
    --model-type match --results-path results/tune_4b.csv \
    --encoder-path results/enc_4b.pt --decoder-path results/dec_4b.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method dot --report-results \
    --model-type match --results-path results/tune_4c.csv \
    --encoder-path results/enc_4c.pt --decoder-path results/dec_4c.pt

python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method vinyals --report-results \
    --model-type match --results-path results/tune_5a.csv \
    --encoder-path results/enc_5a.pt --decoder-path results/dec_5a.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method vinyals --report-results \
    --model-type match --results-path results/tune_5b.csv \
    --encoder-path results/enc_5b.pt --decoder-path results/dec_5b.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.1 --attn-method vinyals --report-results \
    --model-type match --results-path results/tune_5c.csv \
    --encoder-path results/enc_5c.pt --decoder-path results/dec_5c.pt

python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.3 --attn-method luong --report-results \
    --model-type match --results-path results/tune_6a.csv \
    --encoder-path results/enc_6a.pt --decoder-path results/dec_6a.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.3 --attn-method luong --report-results \
    --model-type match --results-path results/tune_6b.csv \
    --encoder-path results/enc_6b.pt --decoder-path results/dec_6b.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 25000 \
    --learning-rate 0.01 --drop-prob 0.3 --attn-method luong --report-results \
    --model-type match --results-path results/tune_6c.csv \
    --encoder-path results/enc_6c.pt --decoder-path results/dec_6c.pt

python 4_attention.py --task-name challenge --decay-times 2 --n-iters 30000 \
    --learning-rate 0.003 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_7a.csv \
    --encoder-path results/enc_7a.pt --decoder-path results/dec_7a.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 30000 \
    --learning-rate 0.003 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_7b.csv \
    --encoder-path results/enc_7b.pt --decoder-path results/dec_7b.pt
python 4_attention.py --task-name challenge --decay-times 2 --n-iters 30000 \
    --learning-rate 0.003 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_7c.csv \
    --encoder-path results/enc_7c.pt --decoder-path results/dec_7c.pt

python 4_attention.py --task-name challenge --decay-times 3 --n-iters 30000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_8a.csv \
    --encoder-path results/enc_8a.pt --decoder-path results/dec_8a.pt
python 4_attention.py --task-name challenge --decay-times 3 --n-iters 30000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_8b.csv \
    --encoder-path results/enc_8b.pt --decoder-path results/dec_8b.pt
python 4_attention.py --task-name challenge --decay-times 3 --n-iters 30000 \
    --learning-rate 0.01 --drop-prob 0.2 --attn-method luong --report-results \
    --model-type match --results-path results/tune_8c.csv \
    --encoder-path results/enc_8c.pt --decoder-path results/dec_8c.pt