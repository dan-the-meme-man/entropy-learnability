#!/bin/bash
#SBATCH --job-name="lrbs"
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

python main.py -d long_range_bigrams -v 10 -s

python main.py -d long_range_bigrams -v 25 -s

python main.py -d long_range_bigrams -v 50 -s

python main.py -d long_range_bigrams -v 75 -s

python main.py -d long_range_bigrams -v 100 -s

python main.py -d long_range_bigrams -v 250 -s

python main.py -d long_range_bigrams -v 500 -s

python main.py -d long_range_bigrams -v 750 -s

python main.py -d long_range_bigrams -v 1000 -s

python main.py -d long_range_bigrams -v 2500 -s

python main.py -d long_range_bigrams -v 5000 -s

python main.py -d long_range_bigrams -v 7500 -s

python main.py -d long_range_bigrams -v 10000 -s
