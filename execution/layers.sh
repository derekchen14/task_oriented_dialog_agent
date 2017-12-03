
#!/usr/bin/env bash
#python 4_attention.py --task-name dstc --hidden-size 512 --n-layers 2 --drop-prob 0.0 -enp 'results/2x512en.pt' -edp 'results/2x512_de.pt' -ep 'results/2x512_error.csv' -s -teach 0.5

#python 4_attention.py --task-name dstc --hidden-size 1024 --n-layers 2 --drop-prob 0.0 -enp 'results/2x1024en.pt' -edp 'results/2x1024_de.pt' -ep 'results/2x1024_error.csv' -s -teach 0.5


python 4_attention.py --task-name dstc --hidden-size 512 --n-layers 2 --drop-prob 0.0 -enp 'results/2x512_05en.pt' -edp 'results/2x512_05de.pt' -ep 'results/2x512_error05.csv' -s -teach 0.5

python 4_attention.py --task-name dstc --hidden-size 1024 --n-layers 2 --drop-prob 0.0 -enp 'results/2x1024_05en.pt' -edp 'results/2x1024_05de.pt' -ep 'results/2x1024_error05.csv' -s -teach 0.5

python 4_attention.py --task-name dstc --hidden-size 512 --n-layers 2 --drop-prob 0.0 -enp 'results/2x1024_075en.pt' -edp 'results/2x1024_075de.pt' -ep 'results/2x1024_error05.csv' -s -teach 0.75

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 3 --drop-prob 0.0 -enp 'results/3layer_en.pt' -edp 'results/3_layer2_de.pt' -ep 'results/3layer2_error.csv' -s -teach 0.5
