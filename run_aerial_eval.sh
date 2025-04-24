#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000

module load anaconda/3

conda activate rs_fairness

python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Aerial_SwinB_SI" --saved_weights "Satlas_Aerial_SwinB_SI_sgd_lr090_model_weights.pth" --split "random"
python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Aerial_SwinB_SI" --saved_weights "Satlas_Aerial_SwinB_SI_sgd_lr090_model_weights.pth" --split "urban_rural"
python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Aerial_SwinB_SI" --saved_weights "Satlas_Aerial_SwinB_SI_sgd_lr0100_model_weights.pth"