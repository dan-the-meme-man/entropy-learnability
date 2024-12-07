#!/bin/bash
#SBATCH --job-name="nb"
#SBATCH --output="%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

module load cuda/12.5
module load gcc/11.4.0

nvidia-smi

python main.py -d normal_bigrams -v 10 -l

python main.py -d normal_bigrams -v 25 -l

python main.py -d normal_bigrams -v 50 -l

python main.py -d normal_bigrams -v 75 -l

python main.py -d normal_bigrams -v 100 -l

python main.py -d normal_bigrams -v 250 -l

python main.py -d normal_bigrams -v 500 -l

python main.py -d normal_bigrams -v 750 -l

python main.py -d normal_bigrams -v 1000 -l

python main.py -d normal_bigrams -v 2500 -l

python main.py -d normal_bigrams -v 5000 -l

python main.py -d normal_bigrams -v 7500 -l

python main.py -d normal_bigrams -v 10000 -l
