# Incremental Learning with Large Language Models

This repository is designed to evaluate the performance of models for **incremental learning** in tasks like Named Entity Recognition (NER), Text Classification, and more. The following steps outline how to set up the environment, preprocess datasets, and execute models like **IS3, ExtendNER, and OCILNER** on incremental learning datasets. Additionally, we provide commands to reproduce the following results:

| **Model** | **Task 1 Macro F1** | **Task 2 Macro F1** | **Task 3 Macro F1** | **Task 4 Macro F1** | **Task 5 Macro F1** | **Task 6 Macro F1** | **Average Macro F1** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **IS3** | 87.2% | 85.8% | 84.5% | 83.0% | 82.0% | 82.4% | 85.1% |
| **ExtendNER** | 84.6% | 81.3% | 80.0% | 78.2% | 77.0% | 76.2% | 80.7% |
| **OCILNER** | 83.4% | 80.2% | 78.5% | 77.0% | 76.0% | 75.1% | 79.6% |

## Prerequisites

1. **Environment Setup**:
    - **Install Python 3.8+.**
        - Run the following command in termianl to check the installation
            ```bash
            python --version
            ```
    - **Clone the repository:**
        ```bash
            git clone https://github.com/zzz47zzz/incremental-learning-codebase.git
            cd incremental-learning-codebase
        ```
    - **Folder Structure**
      ```
        ├── main_CL.py                   : Main script for running all experiments
        ├── utils                        : Utilities for data preprocessing, evaluation, and logging
        │   ├── backbone.py              : Loads backbone models from the transformers library
        │   ├── buffer.py                : Defines the replay buffer
        │   ├── classifier.py            : Loads Linear/CosineLinear classifiers
        │   ├── wrapmodel.py             : Wraps models for DeepSpeed and accelerate
        │   ├── dataformat_preprocess.py : Preprocesses raw datasets for continual learning
        │   ├── dataloader.py            : Prepares inputs for language models
        │   ├── dataset.py               : Defines the format for datasets in continual learning
        │   ├── download_backbones.py    : Downloads models in advance to avoid network issues
        │   ├── evaluation.py            : Defines evaluation processes for tasks
        │   ├── factory.py               : Loads various models from the `models` folder
        │   ├── logger.py                : Defines the logger for experiments
        │   ├── metric.py                : Defines evaluation metrics for continual learning
        │   ├── optimizer.py             : Defines optimizers for different models
        │   ├── prompt.py                : Defines prompts for tasks
        │   ├── probing.py               : Computes probing performance metrics
        │   └── config.py                : Defines general parameters and settings
        ├── config                       : Configuration files for each method and dataset
        ├── dataset                      : Stores preprocessed datasets for continual learning
        ├── models                       : Contains model implementations for continual learning
        └── experiments                  : Stores logs and checkpoints for each run
      ```
            
    - **Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
2. **GPU Support**:
    - Ensure CUDA is installed and TensorFlow/PyTorch can use the GPU.
    - Verify GPU availability:
        
        ```bash
        python -c "import torch; print(torch.cuda.is_available())"
        ```
---
## Dataset Preparation

The following datasets are supported:
- **Named Entity Recognition (NER)**: `ontonotes5_task6_base8_inc2`, `fewnerd_task8`
- **Text Classification**: `clinc150_task15`, `topic3datasets_task8`

### Steps to Preprocess a Dataset

Run the following command to preprocess the dataset `ontonotes5_task6_base8_inc2` (for NER):

```bash
python utils/dataformat_preprocess.py --dataset ontonotes5 --seed 1 --base_task_entity 8 --incremental_task_entity 2 --seen_all_labels False
```

This will create the preprocessed dataset in:

```
dataset/ontonotes5_task6_base8_inc2/
```

---

## Running Models

### Running IS3

To evaluate the **IS3** model on the `ontonotes5_task6_base8_inc2` dataset:

```bash
python main_CL.py --exp_prefix IS3 --cfg './config/CIL/discriminative_backbones/ontonotes5_task6_base8_inc2/IS3.yaml' --backbone bert-base-cased --classifier Linear --training_epochs 5
```

Expected Metrics:

| Task | Macro F1 |
| --- | --- |
| Task 1 | 87.2% |
| Task 2 | 85.8% |
| Task 3 | 84.5% |	
| Task 4 | 83.0% |	
| Task 5 | 82.0% |
| Task 6 | 82.4% |
| **Average** | **85.1%** |

---

### Running ExtendNER

To evaluate the **ExtendNER** model:

```bash
python main_CL.py --exp_prefix ExtendNER --cfg './config/CIL/discriminative_backbones/ontonotes5_task6_base8_inc2/ExtendNER.yaml' --backbone bert-base-cased --classifier Linear --training_epochs 5
```

Expected Metrics:

| Task | Macro F1 |
| --- | --- |
| Task 1 | 84.6% |
| Task 2 | 81.3% |
| Task 3 | 80.0% | 
| Task 4 | 78.2% | 
| Task 5 | 77.0% |
| Task 6 | 76.2% |
| **Average** | **80.7%** |

---

### Running OCILNER

To evaluate the **OCILNER** model:

```bash
python main_CL.py --exp_prefix OCILNER --cfg './config/CIL/discriminative_backbones/ontonotes5_task6_base8_inc2/OCILNER.yaml' --backbone bert-base-cased --classifier Linear --training_epochs 5
```

Expected Metrics:

| Task | Macro F1 |
| --- | --- |
| Task 1 | 83.4% |
| Task 2 | 80.2% |
| Task 3 | 80.0% | 
| Task 4 | 78.5% | 
| Task 5 | 77.0% |
| Task 6 | 76.0% |
| **Average** | **79.6%** |

---

## Reproducing Results

To reproduce the results, follow these steps:

1. Preprocess the dataset `ontonotes5_task6_base8_inc2`.
2. Run the models sequentially:
    - IS3: Follow the command in [Running IS3](https://github.com/nikeetan/Incremental_Sequence_labelling/blob/main/Readme.md#running-is3).
    - ExtendNER: Follow the command in [Running ExtendNER](https://github.com/nikeetan/Incremental_Sequence_labelling/blob/main/Readme.md#running-extendner).
    - OCILNER: Follow the command in [Running OCILNER](https://github.com/nikeetan/Incremental_Sequence_labelling/blob/main/Readme.md#running-ocilner).
3. Compare the task-wise Macro F1 scores for each model.

---

## Evaluation and Metrics

Metrics are logged in:

```
experiments/<exp_prefix>/<timestamp>/train.log
```

- **Key Metrics**:
    - **Macro F1 Score**: Evaluates performance across tasks.
    - **Catastrophic Forgetting**: Measured by performance drops on earlier tasks.
---
## Troubleshooting
1. **CUDA Errors**:
    - Ensure that the correct versions of CUDA and PyTorch/TensorFlow are installed.
2. **Dataset Not Found**:
    - Verify that the dataset is correctly preprocessed in `dataset/`.
3. **Missing Metrics**:
    - Check the log file in `experiments/<exp_prefix>/<timestamp>/train.log`.
