python 4_attention.py --task-name dstc --hidden-size 512 --n-layers 2 --drop-prob 0.0 -enp 'results/2x512_en.pt' -edp 'results/2x512_de.pt' -ep 'results/2x512_error.csv' -s -teach 0.25

python 4_attention.py --task-name dstc --hidden-size 1024 --n-layers 2 --drop-prob 0.0 -enp 'results/2x1024_en.pt' -edp 'results/2x1024_de.pt' -ep 'results/2x1024_error.csv' -s -teach 0.25
