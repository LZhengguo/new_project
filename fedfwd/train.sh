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

### fwd aggregate fwdgrad
python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
        --n_parties 4 --cls_num 10 --device cuda:3 \
        --comm_round 40  --test_round 30 --sample 1 \
        --dataset office --lr 0.001 --epochs 1 --batch_size 10\
        --check_layer_id 2 --var_threshold 0.1  --k 5 --perturb_num 20\
        --fwdtrain_grad \
        --peftmode adapter
#         --peftmode prompt_tuning 
#         --peftmode adapter 
#         --peftmode lora
#         --peftmode bitfit
### fwd aggregate params
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#         --n_parties 4 --cls_num 10 --device cuda:3 \
#         --comm_round 40  --test_round 30 --sample 1 \
#         --dataset office --lr 1 --epochs 10 --batch_size 10\
#         --check_layer_id 2 --var_threshold 0.1  --k 5 \
#         --fwdtrain_param \
#         --peftmode adapter
        # --peftmode prompt_tuning 
        # --peftmode adapter 
        # --peftmode lora
        # --peftmode bitfit
        # --bpfirst

### bp test
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#           --n_parties 4 --cls_num 10 --device cuda:3 \
#          --batch_size 10 --comm_round 70  --test_round 30 --sample 1 \
#         --dataset office --lr 0.01 --epochs 10\
#         --peftmode adapter --check_layer_id 20 --bptrain --var_threshold 0.1 