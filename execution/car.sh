# python 4_attention.py -t schedule --hidden-size 256 --n-layers 1 --drop-prob 0.3 -enp 'results/sch1_en.pt' -edp 'results/sch1_de.pt' -ep 'results/sch1_error.csv' -s -teach 0.60

# python 4_attention.py -t schedule --hidden-size 256 --n-layers 1 --drop-prob 0.3 -enp 'results/sch2_en.pt' -edp 'results/sch2_de.pt' -ep 'results/sch2_error.csv' -s -teach 0.60

# python 4_attention.py -t weather --hidden-size 256 --n-layers 1 --drop-prob 0.3 -enp 'results/we1_en.pt' -edp 'results/we1_de.pt' -ep 'results/we1_error.csv' -s -teach 0.60

# python 4_attention.py -t weather --hidden-size 256 --n-layers 1 --drop-prob 0.3 -enp 'results/we2_en.pt' -edp 'results/we2_de.pt' -ep 'results/we2_error.csv' -s -teach 0.60

# python 4_attention.py -t navigate --hidden-size 256 --n-layers 1 --drop-prob 0.3 -enp 'results/na1_en.pt' -edp 'results/na1_de.pt' -ep 'results/na1_error.csv' -s -teach 0.60

# python 4_attention.py -t navigate --hidden-size 256 --n-layers 1 --drop-prob 0.3 -enp 'results/na2_en.pt' -edp 'results/na2_de.pt' -ep 'results/na2_error.csv' -s -teach 0.60


python 4_attention.py -t schedule --hidden-size 256 --n-layers 2 --drop-prob 0.3 -enp \
'results/sch1_2l_en.pt' -edp 'results/sch1_2l_de.pt' -ep 'results/sch1_2l_error.csv' -s -teach 0.60

python 4_attention.py -t schedule --hidden-size 256 --n-layers 1 --drop-prob 0.3 -enp -w 0.001\
'results/sch1_wd_en.pt' -edp 'results/sch1_wd_de.pt' -ep 'results/sch1_wd_error.csv' -s -teach 0.60

python 4_attention.py -t schedule --hidden-size 256 --n-layers 2 --drop-prob 0.3 -enp -w 0.001\
'results/sch1_wd2_en.pt' -edp 'results/sch1_wd2_de.pt' -ep 'results/sch1_wd2_error.csv' -s -teach 0.60

python 4_attention.py -t schedule --hidden-size 256 --n-layers 1 --drop-prob 0.1 -enp -w 0.001\
'results/sch1_dp1_en.pt' -edp 'results/sch1_dp1_de.pt' -ep 'results/sch1_dp1_error.csv' -s -teach 0.60

python 4_attention.py -t schedule --hidden-size 256 --n-layers 1 --drop-prob 0.5 -enp -w 0.001\
'results/sch1_dp2_en.pt' -edp 'results/sch1_dp2_de.pt' -ep 'results/sch1_dp2_error.csv' -s -teach 0.60