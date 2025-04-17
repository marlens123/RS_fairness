#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB

module load anaconda/3

conda activate rs_fairness

export WANDB_API_KEY=d2ca547c9f807e8db70308537f4d7b64b6077b81

python -m src.eval --pretraining_dataset "none" --imagenet_model_identifier "swinb" --saved_weights "scratch_sgd_lr029_model_weights.pth"
python -m src.eval --pretraining_dataset "none" --imagenet_model_identifier "swinb" --saved_weights "scratch_sgd_lr029_model_weights.pth" --split "random"
python -m src.eval --pretraining_dataset "none" --imagenet_model_identifier "swinb" --saved_weights "scratch_sgd_lr029_model_weights.pth" --split "urban_rural"