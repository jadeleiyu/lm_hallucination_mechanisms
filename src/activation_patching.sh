#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --array=1,2
#SBATCH --account=def-yangxu
#SBATCH --output=./log/%A_%a.out
#SBATCH --error=./log/%A_%a.err
source /home/leiyu/py3/bin/activate
python activation_patching.py $SLURM_ARRAY_TASK_ID