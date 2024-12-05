'''
"""
This script evaluates the in-context learning performance of GPT-3.5 and GPT-4 on the Concept-1K dataset.
The script performs the following steps:
1. Sets up the environment and imports necessary libraries.
2. Defines prompts for 1-shot and 5-shot learning scenarios.
3. Iterates over different models (GPT-3.5 and GPT-4) and different configurations (1-shot and 5-shot, with different methods).
4. Loads the dataset and selects a subset of samples for evaluation.
5. Constructs the input prompts based on the selected method (same_instance, same_concept, or random).
6. Sends the prompts to the OpenAI API and collects the responses.
7. Compares the responses with the ground truth and calculates the accuracy.
8. Logs the progress and results to a file.
The script uses the following variables:
- `seed`: Random seed for reproducibility.
- `client`: OpenAI API client.
- `role`: Role description for the assistant.
- `prompt_1shot` and `prompt_5shot`: Templates for constructing input prompts.
- `n_subsample`: Number of samples to evaluate.
- `model_name`: Name of the model to evaluate.
- `shot`: Number of shots (1 or 5).
- `method`: Method for selecting training examples (same_instance, same_concept, or random).
- `save_name`: Name for saving the results.
- `cur_dir`: Current directory of the script.
- `train_data` and `test_data`: Training and testing data from the dataset.
- `select_index`: Indices of the selected samples.
- `cnt`: Counter for the number of evaluated samples.
- `acc_list`: List of accuracies for each evaluated sample.
- `prompted_test_input`: Constructed input prompt for the model.
- `test_target`: Ground truth answer for the test sample.
- `completion`: Response from the OpenAI API.
- `response`: Processed response from the model.
"""
This file is for evaluating the in-context learning performance of GPT3.5 and GPT4 on Concept-1K
'''

from openai import OpenAI
import json
import os
import random
import numpy as np

seed = 20240210
random.seed(seed)

client = OpenAI(api_key="")
role = "You're a helpful assistant."

# ============================================================================
prompt_1shot = "I will provide some knowledge as follows:\n \
Question: {0}\n Short Answer: {1}\n \
Please answer the following question according to the above knowledge:\n \
Question: {2}\n Short Answer: "

prompt_5shot = "I will provide some knowledge as follows:\n \
Question: {0}\n Short Answer: {1}\n \
Question: {2}\n Short Answer: {3}\n \
Question: {4}\n Short Answer: {5}\n \
Question: {6}\n Short Answer: {7}\n \
Question: {8}\n Short Answer: {9}\n \
Please answer the following question according to the above knowledge:\n \
Question: {10}\n Short Answer: "
# ============================================================================

n_subsample = 500

# model_name = 'gpt-3.5-turbo' # gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo 
# shot = 1
# method = 'same_concept' # same_instance, same_concept, rand

for model_name in ['gpt-3.5-turbo','gpt-4-turbo-preview']:
    for shot in [1,5]:
        for method in ['rand','same_concept']:
            if model_name == 'gpt-3.5-turbo' and shot == 1 and method=='rand':
                continue

            save_name = '%dshot_%s'%(shot,method)

            cur_dir = os.path.dirname(__file__)
            with open(os.path.join(cur_dir,'dataset/concept_1k_task1/continual_data.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            train_data = data['0']['train']
            test_data = data['0']['test']
            select_index = list(range(len(train_data['input'])))
            random.shuffle(select_index)
            select_index = select_index[:n_subsample]

            cnt = 0
            acc_list = []
            for test_sample_id in select_index:
                if shot == 1:
                    if method == 'same_instance':
                        prompted_test_input = prompt_1shot.format(
                            train_data['input'][test_sample_id],
                            train_data['target'][test_sample_id],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'same_concept':
                        same_concept_idx = np.where(np.array(train_data['concept_id'])==test_data['concept_id'][test_sample_id])[0]
                        random_index = np.random.choice(same_concept_idx,size=1)
                        prompted_test_input = prompt_1shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'rand':
                        random_index = np.random.choice(list(range(len(train_data['input']))),size=1)
                        prompted_test_input = prompt_1shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            test_data['input'][test_sample_id],
                        )
                    else:
                        raise NotImplementedError()
                elif shot==5:
                    if method == 'same_instance':
                        random_index = np.random.choice(list(range(len(train_data['input']))),size=4)
                        prompted_test_input = prompt_5shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            train_data['input'][random_index[1]],
                            train_data['target'][random_index[1]],
                            train_data['input'][random_index[2]],
                            train_data['target'][random_index[2]],
                            train_data['input'][random_index[3]],
                            train_data['target'][random_index[3]],
                            train_data['input'][test_sample_id],
                            train_data['target'][test_sample_id],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'same_concept':
                        same_concept_idx = np.where(np.array(train_data['concept_id'])==test_data['concept_id'][test_sample_id])[0]
                        random_index = np.random.choice(same_concept_idx,size=5)
                        prompted_test_input = prompt_5shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            train_data['input'][random_index[1]],
                            train_data['target'][random_index[1]],
                            train_data['input'][random_index[2]],
                            train_data['target'][random_index[2]],
                            train_data['input'][random_index[3]],
                            train_data['target'][random_index[3]],
                            train_data['input'][random_index[4]],
                            train_data['target'][random_index[4]],
                            test_data['input'][test_sample_id],
                        )
                    elif method == 'rand':
                        random_index = np.random.choice(list(range(len(train_data['input']))),size=5)
                        prompted_test_input = prompt_5shot.format(
                            train_data['input'][random_index[0]],
                            train_data['target'][random_index[0]],
                            train_data['input'][random_index[1]],
                            train_data['target'][random_index[1]],
                            train_data['input'][random_index[2]],
                            train_data['target'][random_index[2]],
                            train_data['input'][random_index[3]],
                            train_data['target'][random_index[3]],
                            train_data['input'][random_index[4]],
                            train_data['target'][random_index[4]],
                            test_data['input'][test_sample_id],
                        )
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

                test_target = test_data['target'][test_sample_id]
                        
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": prompted_test_input}
                    ]
                )

                response = completion.choices[0].message.content
                response = response.strip().lower()
                test_target = test_target.strip().lower()
                acc_list.append(1.0 if response==test_target else 0.0)
                cnt += 1

                print('Progress %.2f%%[=%d/%d]: Test_ID=%s, Target=%s, Predict=%s, tmp_ACC=%.2f%%'%(
                    cnt/len(select_index)*100,
                    cnt,
                    len(select_index),
                    test_sample_id,
                    test_target,
                    response,
                    np.mean(acc_list)*100,
                ))

                with open(os.path.join(cur_dir,'./seed%d_%s_%s.txt'%(seed,model_name,save_name)), 'a') as file:
                    file.write('Progress %.2f%%[=%d/%d]: Test_ID=%s, Target=%s, Predict=%s, tmp_ACC=%.2f%%\n'%(
                    cnt/len(select_index)*100,
                    cnt,
                    len(select_index),
                    test_sample_id,
                    test_target,
                    response,
                    np.mean(acc_list)*100,
                ))


