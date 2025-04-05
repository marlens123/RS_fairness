#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB

module load anaconda/3

conda activate rs_fairness

export WANDB_API_KEY=d2ca547c9f807e8db70308537f4d7b64b6077b81

python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Aerial_SwinB_SI_RGB" --saved_weights "Satlas_Aerial_SwinB_SI_RGB_sgd_lr075_model_weights.pth"