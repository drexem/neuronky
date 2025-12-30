#!/bin/bash
#SBATCH -J neuronky_mlp
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -o main%j.out
#SBATCH -e main%j.err

# Activate your virtual environment
source neuronky/bin/activate

# Navigate to project directory
cd /home/kolcunm/neuronky


export PYTHONPATH=/home/kolcunm/neuronky

python main.py