#!/usr/bin/env bash
python 4_attention.py -t challenge --hidden-size 256 --n-layers 1 --drop-prob 0.0 -ep 'results/bid1_error.csv' -teach 0.60

python 4_attention.py -t challenge --hidden-size 256 --n-layers 2 --drop-prob 0.0 -ep 'results/bid2_error.csv' -teach 0.60

python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.0 -ep 'results/bid3_error.csv' -teach 0.60

python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.25 -ep 'results/bid4_error.csv' -teach 0.60

python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.5 -ep 'results/bid5_error.csv' -teach 0.60

