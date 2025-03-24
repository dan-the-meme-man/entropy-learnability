#!/bin/bash
#SBATCH --job-name="nb"
#SBATCH --output="%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

source ~/.bashrc

module load cuda/12.5
module load gcc/11.4.0

nvidia-smi

conda deactivate
conda deactivate
conda activate entropy

python main.py -d normal_bigrams -v 10 --do 0.0

python main.py -d normal_bigrams -v 25 --do 0.0

python main.py -d normal_bigrams -v 50 --do 0.0

python main.py -d normal_bigrams -v 75 --do 0.0

python main.py -d normal_bigrams -v 100 --do 0.0

python main.py -d normal_bigrams -v 250 --do 0.0

python main.py -d normal_bigrams -v 500 --do 0.0

python main.py -d normal_bigrams -v 750 --do 0.0

python main.py -d normal_bigrams -v 1000 --do 0.0

python main.py -d normal_bigrams -v 2500 --do 0.0

python main.py -d normal_bigrams -v 5000 --do 0.0

python main.py -d normal_bigrams -v 7500 --do 0.0

python main.py -d normal_bigrams -v 10000 --do 0.0
