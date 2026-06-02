<div align="center">

# RASO

### Reference-Free Adversarial Sex Obfuscation in Speech

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-11249000-blue)](https://ieeexplore.ieee.org/document/11249000)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FAPSIPAASC65261.2025.11249000-green)](https://doi.org/10.1109/APSIPAASC65261.2025.11249000)
[![Conference](https://img.shields.io/badge/APSIPA%20ASC-2025-orange)](https://ieeexplore.ieee.org/document/11249000)

</div>

## Overview

RASO is a reference-free speech privacy method for sex obfuscation. It aims to suppress sex-discriminative acoustic cues in speech while preserving linguistic content, without requiring target-speaker reference speech.

The method combines adversarial learning, acoustic feature manipulation, F0/formant-oriented regularisation, neural vocoder-based waveform reconstruction, and external attacker-based evaluation for sex leakage.

Paper link:
- IEEE Xplore: https://ieeexplore.ieee.org/document/11249000


## Repository Structure

```text
.
├── train.py                         # Main training entry
├── model.py                         # RASO model architecture
├── data_loader.py                   # Data loading utilities
├── config.py                        # General configuration
├── vocoder_config.py                # HiFi-GAN/vocoder configuration
├── synthesize.py                    # Synthesis utilities
├── hifigan_inference.py             # HiFi-GAN inference helper
├── inference_test.py                # Inference/test helper
├── evaluate_conversion.py           # Conversion evaluation helper
├── hifi_gan/                        # HiFi-GAN vocoder code
├── evaluation/
│   ├── eval_audeering.py            # Audeering-based sex leakage evaluation
│   ├── eval_audeering_stargan.py    # Historical audeering evaluation variant
│   ├── eval_hubert.py               # HuBERT-based historical evaluation
│   ├── eval_hubert_seed.py          # Seeded HuBERT evaluation variant
│   ├── finetune_attacker.py         # Semi-informed attacker fine-tuning
│   ├── prepare_attacker_data.py     # Data preparation for attacker fine-tuning
│   └── gender_model.py              # Gender classifier model wrapper
├── scripts/
│   ├── train.sh
│   ├── eval_audeering.sh
│   └── eval_hubert.sh
├── requirements.txt
└── README.md
```

## Installation

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code was developed with PyTorch, torchaudio, transformers, librosa, scikit-learn, matplotlib, and HiFi-GAN-related dependencies.

## Training

Main entry:

```bash
python train.py
```

or:

```bash
bash scripts/train.sh
```

Before training, update local dataset and vocoder paths in:

- `train.py`
- `data_loader.py`
- `vocoder_config.py`

Large datasets, checkpoints, generated wav files, and TensorBoard logs are not included in this repository.

## Evaluation

Audeering-based sex leakage evaluation:

```bash
bash scripts/eval_audeering.sh
```

HuBERT-based historical evaluation:

```bash
bash scripts/eval_hubert.sh
```

Evaluation scripts are located in:

```text
evaluation/
```

The evaluation code includes scripts for:

- pre-trained attacker evaluation;
- semi-informed attacker fine-tuning;
- DET/EER-based sex leakage analysis;
- score-distribution analysis.

## Checkpoints and Generated Audio

Model checkpoints and generated audio are not stored in this GitHub repository.

Recommended handling:

- keep checkpoints outside GitHub;
- host large model files on Hugging Face or another model hosting service;
- keep generated wav files and intermediate features out of the repository.

## Citation

If you use this code, please cite:

```bibtex
@INPROCEEDINGS{11249000,
  author={Qu, Yangyang and Panariello, Michele and Todisco, Massimiliano and Evans, Nicholas},
  booktitle={2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)}, 
  title={Reference-Free Adversarial Sex Obfuscation in Speech}, 
  year={2025},
  pages={2128-2133},
  doi={10.1109/APSIPAASC65261.2025.11249000}
}
```

## License

Please check the license file before using or redistributing this code. If no license is provided, all rights are reserved by the authors.

## Acknowledgements

This work was conducted at EURECOM.
