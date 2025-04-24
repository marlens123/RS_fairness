#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000

module load anaconda/3

conda activate rs_fairness

python -m src.train --pretraining_dataset "ImageNet" --imagenet_model_identifier "swinb" --config_file "sgd_lr0.01.py" --random_seed 10 --split "random"