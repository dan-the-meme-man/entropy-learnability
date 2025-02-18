#!/bin/bash
#SBATCH --job-name="uu"
#SBATCH --output="%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

module load cuda/12.5
module load gcc/11.4.0

nvidia-smi

conda activate entropy

python main.py -d uniform_unigrams -v 10

python main.py -d uniform_unigrams -v 25

python main.py -d uniform_unigrams -v 50

python main.py -d uniform_unigrams -v 75

python main.py -d uniform_unigrams -v 100

python main.py -d uniform_unigrams -v 250

python main.py -d uniform_unigrams -v 500

python main.py -d uniform_unigrams -v 750

python main.py -d uniform_unigrams -v 1000

python main.py -d uniform_unigrams -v 2500

python main.py -d uniform_unigrams -v 5000

python main.py -d uniform_unigrams -v 7500

python main.py -d uniform_unigrams -v 10000
