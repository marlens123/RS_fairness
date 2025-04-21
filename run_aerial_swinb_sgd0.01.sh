#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000

module load anaconda/3

conda activate rs_fairness

export WANDB_API_KEY=d2ca547c9f807e8db70308537f4d7b64b6077b81

python -m src.train --pretraining_dataset "Satlas" --satlas_model_identifier "Aerial_SwinB_SI" --config_file "sgd_lr0.01.py"