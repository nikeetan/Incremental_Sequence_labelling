config_setting='--config_file your-accelerate-config-file.yaml'

exp_prefix='SOTACILRE'


for i in $(seq 1 3);
do
    echo 'Run '$i


    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/L2KD.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/LAMOL_g.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/LAMOL_t.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/LAMOL_KD.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/PCLL.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='EleutherAI/pythia-160m-deduped'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    classifier='None'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/L2KD.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/LAMOL_g.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/LAMOL_t.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/LAMOL_KD.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/PCLL.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1




    # ========================================================================================================================
    backbone='EleutherAI/pythia-410m-deduped'
    classifier='CosineLinear'
    # ========================================================================================================================

    # ---------------------------------------------------------------------------------------------------------------------------
    wandb_setting='--is_wandb True --wandb_project Class-Incremental-RelationExtraction --wandb_entity your-name'
    # ---------------------------------------------------------------------------------------------------------------------------
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/SEQ_pre_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1
    accelerate launch $config_setting main_CL.py $wandb_setting --exp_prefix $exp_prefix-$backbone-$classifier --cfg './config/CIL/generative_backbones/tacred_task8/SEQ_warm_fix.yaml' --backbone $backbone --classifier $classifier --training_epochs 5 --evaluate_interval -1





done