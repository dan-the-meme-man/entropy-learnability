#!/bin/bash
#SBATCH --job-name="nu_lstm"
#SBATCH --output="%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

source ~/.bashrc

 load cuda/12.5
module load gcc/11.4.0

nvidia-smi

conda deactivate
conda deactivate
conda activate entropy

python main.py -d normal_unigrams -v 10 -l

python main.py -d normal_unigrams -v 25 -l

python main.py -d normal_unigrams -v 50 -l

python main.py -d normal_unigrams -v 75 -l

python main.py -d normal_unigrams -v 100 -l

python main.py -d normal_unigrams -v 250 -l

python main.py -d normal_unigrams -v 500 -l

python main.py -d normal_unigrams -v 750 -l

python main.py -d normal_unigrams -v 1000 -l

python main.py -d normal_unigrams -v 2500 -l

python main.py -d normal_unigrams -v 5000 -l

python main.py -d normal_unigrams -v 7500 -l

python main.py -d normal_unigrams -v 10000 -l
