python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 2 --drop-prob 0.0 -enp 'results/2layer_en.pt' -edp 'results/2layer_de.pt' -ep 'results/2layer_error.csv' -s

python 4_attention.py --task-name dstc --hidden-size 512 --n-layers 1 --drop-prob 0.0 -enp 'results/1layer_en.pt' -edp 'results/1layer_de.pt' -ep 'results/1layer_error.csv' -s
~                                                      
python 4_attention.py --task-name dstc --hidden-size 256 --n-layers 3 --drop-prob 0.0 -enp 'results/3layer_en.pt' -edp 'results/3layer_de.pt' -ep 'results/3layer_error.csv' -s
~                                                       
