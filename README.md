# RASO: Reference-free Adversarial Sex Obfuscation in Speech

This repository contains the cleaned training and evaluation code for RASO.

## Main files

Training entry:
- train.py

Model:
- model.py

Data loader:
- data_loader.py

Vocoder configuration:
- vocoder_config.py

## Training

Run:

    bash scripts/train.sh

Before training, update local dataset and vocoder paths in:

- train.py
- data_loader.py
- vocoder_config.py

## Evaluation

Audeering-based evaluation:

    bash scripts/eval_audeering.sh

HuBERT-based historical evaluation:

    bash scripts/eval_hubert.sh

Evaluation scripts are located in:

- evaluation/

## Notes

Large datasets, generated audio files, checkpoints, and TensorBoard logs are not included in this repository.
