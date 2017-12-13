#!/usr/bin/env bash
#python 4_attention.py -t challenge --hidden-size 256 --n-layers 1 --drop-prob 0.0 -ep 'results/bid1_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 256 --n-layers 2 --drop-prob 0.0 -ep 'results/bid2_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.0 -ep 'results/bid3_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.25 -ep 'results/bid4_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.5 -ep 'results/bid5_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.75 -ep 'results/bid6_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.95 -ep 'results/bid7_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.3 -ep 'results/bidre3_error.csv' -teach 0.60

#python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.5 -ep 'results/bidre5_error.csv' -teach 0.60

python 4_attention.py -t challenge --hidden-size 512 --n-layers 1 --drop-prob 0.3 -enp 'results/bid_best.pt' -edp 'results/bid_best.pt' -ep 'results/bid_best_error.csv' -teach 0.60




