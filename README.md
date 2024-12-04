# DPO-Trainer

 This repository is a PyTorch implementation of **Direct Preference Optimization (DPO)**. This project is intended for learning purposes. The reference materials are the [DPO paper](https://arxiv.org/abs/2305.18290) and [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb).

 The dataset used in the repository is directly coming from [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json).

## Installation

To install the required environment, please use the following command:

```bash
conda env create -f environment.yml
```

## Description of each file

alpaca_template.py: An alpaca template file directly from torchtune 2.4.1.

clean_dataset.py: Given the raw dataset with five attributes, the function in the file generate a list of tuples (prompt, chosen, rejected), then split it into training set and testing set.

dataset.py: A PreferenceDataset class for loading preference pairs.

model.py: To load model and tokenizer.

loss.py: DPO loss function.

dpo_trainer.py: main function to train a DPO process.


## Train the model using DPO

To train the model using the DPO loss function, run:

```bash
python dpo_trainer.py
```

The loss_plot.png shows a figure of training and testing loss in terms of epoch.