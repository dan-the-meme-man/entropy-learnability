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

python main.py -d uniform_unigrams -v 10 -f

python main.py -d uniform_unigrams -v 25 -f

python main.py -d uniform_unigrams -v 50 -f

python main.py -d uniform_unigrams -v 75 -f

python main.py -d uniform_unigrams -v 100 -f

python main.py -d uniform_unigrams -v 250 -f

python main.py -d uniform_unigrams -v 500 -f

python main.py -d uniform_unigrams -v 750 -f

python main.py -d uniform_unigrams -v 1000 -f

python main.py -d uniform_unigrams -v 2500 -f

python main.py -d uniform_unigrams -v 5000 -f

python main.py -d uniform_unigrams -v 7500 -f

python main.py -d uniform_unigrams -v 10000 -f
