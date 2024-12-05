# This script preprocesses the dataset using a Python script.

# Run the data format preprocessing script with specified arguments:
# --dataset: Specifies the dataset to be used, in this case, 'ontonotes5'.
# --seed: Sets the random seed for reproducibility, here it is set to 1.
# --base_task_entity: Defines the number of base task entities, set to 8.
# --incremental_task_entity: Defines the number of incremental task entities, set to 2.
# --seen_all_labels: A boolean flag indicating whether all labels have been seen, set to False.
python utils/dataformat_preprocess.py --dataset topic3datasets --seed 1 --num_sample_train_per_class 3000 --num_sample_test_per_class 2000
python utils/dataformat_preprocess.py --dataset clinc150 --seed 1
python utils/dataformat_preprocess.py --dataset banking77 --seed 1
python utils/dataformat_preprocess.py --dataset fewrel --seed 1
python utils/dataformat_preprocess.py --dataset tacred --seed 1
python utils/dataformat_preprocess.py --dataset fewnerd --seed 1 --base_task_entity 6 --incremental_task_entity 6 --seen_all_labels False
python utils/dataformat_preprocess.py --dataset i2b2 --seed 1 --base_task_entity 8 --incremental_task_entity 2 --seen_all_labels False
python utils/dataformat_preprocess.py --dataset ontonotes5 --seed 1 --base_task_entity 8 --incremental_task_entity 2 --seen_all_labels False