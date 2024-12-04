# DPO-Trainer

 This repository is a PyTorch implementation of **Direct Preference Optimization (DPO)**. This project is intended for learning purposes. The reference materials are [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb) and the original [DPO paper](https://arxiv.org/abs/2305.18290).

## Installation

To install the required environment, please use the following command:

```bash
conda env create -f environment.yml
```

To train the model using the DPO loss function, run:

```bash
python dpo_trainer.py
```

The loss_plot.png shows a figure of training and testing loss in terms of epoch.