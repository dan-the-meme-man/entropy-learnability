#!/bin/bash
#SBATCH --job-name="mu"
#SBATCH --output="%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

source ~/.bashrc

module load cuda/12.5
module load gcc/11.4.0

nvidia-smi

conda deactivate
conda deactivate
conda activate entropy

python main.py -d manual_unigrams -m 0.8

python main.py -d manual_unigrams -m 0.7

python main.py -d manual_unigrams -m 0.6

python main.py -d manual_unigrams -m 0.5
