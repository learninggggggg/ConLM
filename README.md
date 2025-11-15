# GEC: A Global and Emotion-Consistent Context Modeling Framework for Suicide Risk Detection

[//]: # (<a href='#'><img src='https://img.shields.io/badge/Paper-PDF-blue'></a>)

[//]: # (<a href='#'><img src='https://img.shields.io/badge/Conference-EMNLP-orange'></a>)
<a href='#'><img src='https://img.shields.io/badge/Language-Python-yellow'></a>
<a href='#'><img src='https://img.shields.io/badge/Framework-PyTorch-red'></a>

[Zhuping Ding](), [Yongpan Sheng](),[Lirong He](), [Yiran Wang](), and [Ming Liu]()

## Release
- [11/25] Initial release of GEC codebase for suicide risk detection.

## Contents
- [Release](#release)
- [Contents](#contents)
- [Overview](#overview)
- [Installation](#installation)

## Overview
GEC is a novel framework for suicide risk detection that incorporates both global context modeling and emotion-consistent representation learning. This approach addresses the critical need for accurate and sensitive detection of suicide risk in online text content.


## Installation
Clone this repository:

```
git clone https://github.com/learninggggggg/GEC-Suicide-Risk-Detection.git
cd GEC-Suicide-Risk-Detection
```

Create and activate a conda environment:
```
conda create -n gec python=3.8 -y
conda activate gec
```
Install required dependencies:

```
pip install -r requirements.txt
```

Install PyTorch (choose the appropriate version for your CUDA setup):
```
# For CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Data Preparation
Download the Reddit dataset and place it in the data/Reddit directory:
```
mkdir -p data/Reddit
wget -O data/Reddit/reddit_clean.pkl https://example.com/reddit_clean.pkl
```
## Training and Evaluation
Run Full Model (Baseline)
```
bash ablation_study.sh
# The full model experiment is executed by default in the script
```
## Run Ablation Experiments
The ablation_study.sh script includes all ablation experiments. You can run specific ones by modifying the script or executing individual commands:
```
# Run without Global Attention
python 消融实验.py \
--mode train_test \
--use_gated_fusion \
--use_ordered_loss \
--loss_type ordered \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/without_global_attention.csv

# Run without Gated Fusion
python 消融实验.py \
--mode train_test \
--use_global_attention \
--use_ordered_loss \
--loss_type ordered \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/without_gated_fusion.csv
```

## Cross-Validation
The code uses 5-fold cross-validation by default. To adjust parameters like epochs or batch size, modify the arguments in the training command:
```
python 消融实验.py \
--mode train_test \
--use_global_attention \
--use_gated_fusion \
--use_ordered_loss \
--loss_type ordered \
--batch_size 32 \
--epochs 50 \
--lr 0.001 \
--data_path data/Reddit/reddit_clean.pkl \
--save_results \
--results_path ablation_results/custom_params.csv
```
## Testing
To test a trained model, use the --mode test flag and specify the model path:
```
python 消融实验.py \
--mode test \
--data_path data/Reddit/reddit_clean.pkl \
--save_path models/best_model.pth \
--results_path test_results.csv
```
## Check Results
Ablation experiment results are saved in the ablation_results directory. You can view them using pandas:
```
import pandas as pd

# View full model results
full_results = pd.read_csv("ablation_results/full_model.csv")
print(full_results)

# Compare all ablation results
ablation_results = [
    pd.read_csv("ablation_results/full_model.csv"),
    pd.read_csv("ablation_results/without_global_attention.csv"),
    pd.read_csv("ablation_results/without_gated_fusion.csv")
]
```
[//]: # (Citation)

[//]: # (If you use GEC in your research, please cite our paper:)

[//]: # ()
[//]: # (bibtex)

[//]: # (@article{ding2024gec,)

[//]: # (  title={GEC: A Global and Emotion-Consistent Context Modeling Framework for Suicide Risk Detection},)

[//]: # (  author={Ding, Zhuping and Sheng, Yongpan and He, Lirong and Wang, Yiran and Liu, Ming},)

[//]: # (  journal={Proceedings of the Conference on Empirical Methods in Natural Language Processing},)

[//]: # (  year={2024})

[//]: # (})

[//]: # (License)

[//]: # (https://img.shields.io/badge/License-MIT-yellow.svg)

[//]: # ()
[//]: # (This project is licensed under the MIT License - see the LICENSE file for details.)

[//]: # ()
[//]: # (Important Note: This software is intended for research purposes only. It should not be used as a substitute for professional mental health advice, diagnosis, or treatment. If you or someone you know is experiencing suicidal thoughts, please contact a mental health professional or emergency services immediately.)

[//]: # ()
[//]: # (Contact)


[//]: # (For questions about this codebase, please open an issue on GitHub or contact [Your Name] at [your.email@institution.edu].)
