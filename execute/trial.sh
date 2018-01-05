#!/usr/bin/env bash
# python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 2 --drop-prob 0.0 -enp 'results/2layer_en.pt'
# -edp 'results/2layer_de.pt' -ep 'results/2layer_error.csv' -s -teach 0.5

# python 4_attention.py --task-name dstc --hidden-size 512 --n-layers 1 --drop-prob 0.0 -enp 'results/1layer_en.pt'
# -edp 'results/1layer_de.pt' -ep 'results/1layer_error.csv' -s
# ~                                                      
# python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 3 --drop-prob 0.0 -enp 'results/3layer_en.pt'
# -edp 'results/3layer_de.pt' -ep 'results/3layer_error.csv' -s
#~

#-----------seq2seq -> attention -> teacher forcing---------------
# seq2seq no lr decay
python 3_seq2seq.py --task-name dstc --hidden-size 256 --n-layers 1 --drop-prob 0.0 -enp 'results/seq_en.pt' -edp 'results/seq_de.pt' -ep 'results/seq_error.csv' -s -teach 0.0

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 1 --drop-prob 0.0 -enp 'results/attn_en.pt' -edp 'results/attn_de.pt' -ep 'results/attn_error.csv' -s -teach 0.0

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 1 --drop-prob 0.0 -enp 'results/t01_en.pt' -edp 'results/t01_de.pt' -ep 'results/t01_error.csv' -s -teach 0.1

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 1 --drop-prob 0.0 -enp 'results/t025_en.pt' -edp 'results/t025_de.pt' -ep 'results/t025_error.csv' -s -teach 0.25

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 1 --drop-prob 0.0 -enp 'results/t05_en.pt' -edp 'results/t05_de.pt' -ep 'results/t05_error.csv' -s -teach 0.5

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 1 --drop-prob 0.0 -enp 'results/t075_en.pt' -edp 'results/t075_de.pt' -ep 'results/t075_error.csv' -s -teach 0.75

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 1 --drop-prob 0.0 -enp 'results/t1_en.pt' -edp 'results/t1_de.pt' -ep 'results/t1_error.csv' -s -teach 1.0



