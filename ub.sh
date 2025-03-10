#!/bin/bash
#SBATCH --job-name="ub"
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

python main.py -d uneven_bigrams -v 10

python main.py -d uneven_bigrams -v 25

python main.py -d uneven_bigrams -v 50

python main.py -d uneven_bigrams -v 75

python main.py -d uneven_bigrams -v 100

python main.py -d uneven_bigrams -v 250

python main.py -d uneven_bigrams -v 500

python main.py -d uneven_bigrams -v 750

python main.py -d uneven_bigrams -v 1000
