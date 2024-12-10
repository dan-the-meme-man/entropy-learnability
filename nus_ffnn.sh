#!/bin/bash
#SBATCH --job-name="nus"
#SBATCH --output="%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

module load cuda/12.5
module load gcc/11.4.0

nvidia-smi

python main.py -d normal_unigrams -v 10 -s -f

python main.py -d normal_unigrams -v 25 -s -f

python main.py -d normal_unigrams -v 50 -s -f

python main.py -d normal_unigrams -v 75 -s -f

python main.py -d normal_unigrams -v 100 -s -f

python main.py -d normal_unigrams -v 250 -s -f

python main.py -d normal_unigrams -v 500 -s -f

python main.py -d normal_unigrams -v 750 -s -f

python main.py -d normal_unigrams -v 1000 -s -f

python main.py -d normal_unigrams -v 2500 -s -f

python main.py -d normal_unigrams -v 5000 -s -f

python main.py -d normal_unigrams -v 7500 -s -f

python main.py -d normal_unigrams -v 10000 -s -f
