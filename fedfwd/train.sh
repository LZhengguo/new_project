## CIFAR100 classes 20
# python main_prompt.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#           --partition noniid-labeluni --n_parties 100 --beta 0.01 --cls_num 10 --device cuda:3 \
#          --batch-size 40 --comm_round 60  --test_round 50 --sample 0.05 --moment 0.5 --rho 0.9 --alg SGPT\
#         --dataset cifar100 --lr 0.01 --epochs 5 --key_prompt 20 --avg_key\
#         --share_blocks 0 1 2 3 --share_blocks_g 4 5 6 

### CIFAR100 classes 50
# python main_prompt.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#           --partition noniid-labeluni --n_parties 100 --beta 0.01 --cls_num 50 --device cuda:4 \
#          --batch-size 40 --comm_round 60  --test_round 50 --sample 0.05 --moment 0.5 --rho 0.9 --alg SGPT \
#         --dataset cifar100 --lr 0.01 --epochs 5 --key_prompt 20 --avg_key\
#         --share_blocks 0 1 2 3 4 --share_blocks_g  5 6 

#########################################
### CIFAR100 classes 50
### bp test
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --partition noniid-labeluni --n_parties 100 --cls_num 50 --beta 0.01 --device cuda:3 \
#         --comm_round 100  --test_round 50 --sample_num 5 \
#         --dataset cifar100 --bp_lr 0.01 --epochs 5 --batch_size 40\
#         --gap_layer 5 \
#         --bptrain \
#         --peftmode adapter
#         --peftmode prompt_tuning 
#         --peftmode adapter 
#         --peftmode lora
#         --peftmode bitfit
#         --test --use_momentum --momentum_f 0.1
### fwdfed new
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --partition noniid-labeluni --n_parties 100 --cls_num 50 --beta 0.01 --device cuda:3 \
#         --comm_round 100  --test_round 50 --sample_num 5 \
#         --dataset cifar100 --epochs 5 --batch_size 40 \
#         --perturb_num 20 --h 0.01 \
#         --gap_layer 5 \
#         --fwdtrain --headbp --m \
#         --peftmode adapter

### fwdfed origin
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --partition noniid-labeluni --n_parties 100 --cls_num 50 --beta 0.01 --device cuda:1 \
#         --comm_round 100  --test_round 50 --sample_num 5 \
#         --dataset cifar100 --epochs 5 --batch_size 40 \
#         --gap_layer 5 \
#         --perturb_num 20 --h 0.01 \
#         --fwdtrain \
#         --peftmode adapter

### fwdllm
python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
        --partition noniid-labeluni --n_parties 100 --cls_num 50 --beta 0.01 --device cuda:2 \
        --comm_round 100  --test_round 50 --sample_num 5 \
        --dataset cifar100 --epochs 5 --batch_size 40 \
        --var_threshold 0.1 --var_control --layer_id_for_check 11 \
        --gap_layer 5 \
        --fwdtrain --Fwdllm \
        --peftmode adapter

#########################################


### CIFAR100 classes 20
### fwd
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --partition noniid-labeluni --n_parties 100 --cls_num 10 --beta 0.01 --device cuda:3 \
#         --comm_round 40  --test_round 30 --sample_num 20 \
#         --dataset cifar100 --lr 0.001 --epochs 1 --batch_size 40\
#         --var_threshold 0.1  --gap_layer 5 --perturb_num 20\
#         --fwdtrain \
#         --peftmode adapter
#         --peftmode prompt_tuning 
#         --peftmode adapter 
#         --peftmode lora
#         --peftmode bitfit
#         --var_control

### bp test
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --partition noniid-labeluni --n_parties 100 --cls_num 10 --beta 0.01 --device cuda:3 \
#         --comm_round 40  --test_round 30 --sample_num 20 \
#         --dataset cifar100 --lr 0.01 --epochs 1 --batch_size 40 \
#         --gap_layer 5 --perturb_num 20 \
#         --bptrain \
#         --peftmode adapter
#         --peftmode prompt_tuning 
#         --peftmode adapter 
#         --peftmode lora
#         --peftmode bitfit



### OFFICE
### fwd
#fwdllm
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --n_parties 4 --cls_num 10 --device cuda:0 \
#         --comm_round 10  --test_round 10 --sample_num 1 \
#         --dataset office --epochs 1 --batch_size 10 \
#         --var_threshold 0.5 --var_control --gap_layer 5 --layer_id_for_check 11 \
#         --fwdtrain --Fwdllm \
#         --peftmode adapter
#         --peftmode prompt_tuning 
#         --peftmode adapter 
#         --peftmode lora
#         --peftmode bitfit

#fwdfed new
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --n_parties 4 --cls_num 10 --device cuda:0 \
#         --comm_round 5  --test_round 5 --sample_num 1 \
#         --dataset office --epochs 5 --batch_size 10\
#         --gap_layer 5 \
#         --perturb_num 20 --h 0.01 --m\
#         --fwdtrain --headbp \
#         --peftmode adapter

#fwdfed origin
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --n_parties 4 --cls_num 10 --device cuda:1 \
#         --comm_round 5  --test_round 5 --sample_num 1 \
#         --dataset office --epochs 1 --batch_size 10\
#         --gap_layer 5 \
#         --perturb_num 20 --h 0.01 \
#         --fwdtrain \
#         --peftmode adapter

### bp test
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --n_parties 4 --cls_num 10 --device cuda:3 \
#         --comm_round 10  --test_round 10 --sample_num 2 \
#         --dataset office --bp_lr 0.01 --epochs 5 --batch_size 10\
#         --gap_layer 5 \
#         --bptrain \
#         --peftmode adapter
#         --peftmode prompt_tuning 
#         --peftmode adapter 
#         --peftmode lora
#         --peftmode bitfit