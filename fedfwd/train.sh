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

# fwd test
python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
          --n_parties 4 --cls_num 10 --device cuda:0 \
         --batch_size 10 --comm_round 70  --test_round 30 --sample 1 --alg SGPT\
        --dataset office --lr 0.001 --epochs 1 -N 20 \
        --peftmode adapter --check_layer_id 20 --fwdtrain_grad --var_threshold 0.1
# --bpfirst

# bp test
python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
          --n_parties 4 --cls_num 10 --device cuda:0 \
         --batch_size 10 --comm_round 70  --test_round 30 --sample 1 --alg SGPT\
        --dataset office --lr 1 --h 0.001 --epochs 10 \
        --peftmode adapter --check_layer_id 20 --bptrain --var_threshold 0.1
# python fwd_main_tc.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#           --n_parties 4 --cls_num 10 --device cuda:3 \
#          --batch_size 50 --comm_round 40  --test_round 30 --sample 1 --alg SGPT\
#         --dataset office --lr 0.01 --epochs 5 \
#         --peftmode adapter