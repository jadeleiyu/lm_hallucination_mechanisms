#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=150G
#SBATCH --account=def-yangxu
#SBATCH --output=./log/%A_%a.out
#SBATCH --error=./log/%A_%a.err
module load gcc/9.3.0 arrow scipy-stack; python -c \"import pyarrow\"
source /home/leiyu/py3/bin/activate
accelerate launch lama_eval.py


export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH