apt-get update
apt-get install nano btop pip -y
conda install nvidia/label/cuda-12.1.0::cuda-nvcc -y
conda install nccl -y
pip install -r requirements.txt
pip install -U numpy