#!/bin/bash
#SBATCH --job-name="nbs_lstm"
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

python main.py -d normal_bigrams -v 10 -s -l

python main.py -d normal_bigrams -v 25 -s -l

python main.py -d normal_bigrams -v 50 -s -l

python main.py -d normal_bigrams -v 75 -s -l

python main.py -d normal_bigrams -v 100 -s -l

python main.py -d normal_bigrams -v 250 -s -l

python main.py -d normal_bigrams -v 500 -s -l

python main.py -d normal_bigrams -v 750 -s -l

python main.py -d normal_bigrams -v 1000 -s -l

python main.py -d normal_bigrams -v 2500 -s -l

python main.py -d normal_bigrams -v 5000 -s -l

python main.py -d normal_bigrams -v 7500 -s -l

python main.py -d normal_bigrams -v 10000 -s -l
