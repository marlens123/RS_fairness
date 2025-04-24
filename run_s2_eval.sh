#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000

module load anaconda/3

conda activate rs_fairness

python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Sentinel2_SwinB_SI_RGB" --saved_weights "Satlas_Sentinel2_SwinB_SI_RGB_sgd_lr090_model_weights.pth" --split "random"
python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Sentinel2_Resnet50_SI_RGB" --saved_weights "Satlas_Sentinel2_Resnet50_SI_RGB_adamw_lr090_model_weights.pth" --split "random"
python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Sentinel2_SwinB_SI_RGB" --saved_weights "Satlas_Sentinel2_SwinB_SI_RGB_sgd_lr090_model_weights.pth" --split "urban_rural"
python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Sentinel2_Resnet50_SI_RGB" --saved_weights "Satlas_Sentinel2_Resnet50_SI_RGB_adamw_lr090_model_weights.pth" --split "urban_rural"
python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Sentinel2_SwinB_SI_RGB" --saved_weights "Satlas_Sentinel2_SwinB_SI_RGB_sgd_lr0100_model_weights.pth"
python -m src.eval --pretraining_dataset "Satlas" --satlas_model_identifier "Sentinel2_Resnet50_SI_RGB" --saved_weights "Satlas_Sentinel2_Resnet50_SI_RGB_adamw_lr0100_model_weights.pth"