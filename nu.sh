#!/bin/bash
#SBATCH --job-name="nu"
#SBATCH --output="%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

module load cuda/12.5
module load gcc/11.4.0

nvidia-smi

python main.py -d normal_unigrams -v 10

python main.py -d normal_unigrams -v 100

python main.py -d normal_unigrams -v 1000

python main.py -d normal_unigrams -v 10000
