#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000

module load anaconda/3

conda activate rs_fairness

python -m src.train --pretraining_dataset "Satlas" --satlas_model_identifier "Aerial_SwinB_SI" --config_file "adamw_lr0.0005.py"