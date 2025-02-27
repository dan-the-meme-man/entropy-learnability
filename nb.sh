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

python main.py -d normal_bigrams -v 10

python main.py -d normal_bigrams -v 25

python main.py -d normal_bigrams -v 50

python main.py -d normal_bigrams -v 75

python main.py -d normal_bigrams -v 100

python main.py -d normal_bigrams -v 250

python main.py -d normal_bigrams -v 500

python main.py -d normal_bigrams -v 750

python main.py -d normal_bigrams -v 1000

python main.py -d normal_bigrams -v 2500

python main.py -d normal_bigrams -v 5000

python main.py -d normal_bigrams -v 7500

python main.py -d normal_bigrams -v 10000
