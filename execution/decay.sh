python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 2 --drop-prob 0.0 -enp 'results/w0_en.pt' -edp 'results/w0_de.pt' -ep 'results/w0_error.csv' -s -teach 0.5 -wd 0.0

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 2 --drop-prob 0.0 -enp 'results/w00001_en.pt' -edp 'results/w00001_de.pt' -ep 'results/w00001_error.csv' -s -teach 0.5 -wd 0.0001

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 2 --drop-prob 0.0 -enp 'results/w0001_en.pt' -edp 'results/w0001_de.pt' -ep 'results/w0001_error.csv' -s -teach 0.5 -wd 0.001

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 2 --drop-prob 0.0 -enp 'results/w001_en.pt' -edp 'results/w001_de.pt' -ep 'results/w001_error.csv' -s -teach 0.5 -wd 0.01

python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 2 --drop-prob 0.0 -enp 'results/w01_en.pt' -edp 'results/w01_de.pt' -ep 'results/w01_error.csv' -s -teach 0.5 -wd 0.1

