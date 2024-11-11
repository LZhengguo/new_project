
import numpy as np
import json
import torch
from collections import  defaultdict
from pathlib import Path
import os
import copy
from math import *
import random
from modelinit import *
import numpy as np
from office_dataset import prepare_data
from domainnet_dataset import prepare_data_domain
from utils import *
import argparse
from models.vit_models_fwd import *
import time

import logging

# logging.basicConfig(
#     filename='logs/train.log',
#     level=print,     
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='ViT-B_16', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-labeluni', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=60, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--test_round', type=int, default=50)  
    parser.add_argument('--root_path', type=str, default='', help='Noise type: None/increasng/space')
    parser.add_argument('--var_control', action='store_true', help="whether use var to control perturb sample times")
    parser.add_argument('--var_threshold', type=float, default=0.5, help='the threshold of perturb variance')
    parser.add_argument('--gap_layer', type=int, default=5, help='per k layer apply one adpater or lora block')
    parser.add_argument('--perturb_num', type=int, default=10, help='num of noise in fwd perturb')
    parser.add_argument('--sample_num', type=int, help='num of random clients participating train per round')
    """
    Used for model 
    """
    parser.add_argument('--model_type', type=str, default='ViT-B_16')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained_dir', type=str, default=None, help='The pretrain model path')
    parser.add_argument('--cls_num', type=int, default=10) 
    parser.add_argument('--fwdtrain',action='store_true', help='Use forward gradient to update parameters')
    parser.add_argument('--bptrain',action='store_true')
    parser.add_argument('--peftmode', type=str, help='set peft mode in (adapter、lora、bitfit)')
    parser.add_argument('--bpfirst',action='store_true')
    parser.add_argument('--lora_alpha',type=float,default=1, help='scaling factor of lora config')
    parser.add_argument('--lora_rank',type=int,default=4, help='lora rank')
    parser.add_argument('--n_prompt',type=int,default=1, help='the num of tokens appended before prompt (prompt tuning)')
    
    args = parser.parse_args()
    return args
args = get_args()

if args.n_parties < args.sample_num:
    print("Error:The sample num is larger than exact worker num!")
    exit()


if args.dataset == 'CIFAR-100':
    cls_coarse = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
    
save_path = args.model_type+"-"+str(args.n_parties)+"-"+args.dataset+'-'+str(args.cls_num)+"-"+args.partition
root_path = args.logdir

save_path = Path(os.path.join(root_path,save_path))
save_path.mkdir(parents=True, exist_ok=True)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
print(save_path)

with open(os.path.join(save_path,'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

### dataloder init
if args.dataset not in ['office','domainnet']:
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
                args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir,args=args)

if args.dataset == 'office':
    data_loader_dict,net_dataidx_map_train = prepare_data(args)
    num_classes = 10
elif args.dataset == 'domainnet':
    data_loader_dict,net_dataidx_map_train = prepare_data_domain(args)
    num_classes = 10
else:
    ###### Data Set related ###### 
    data_loader_dict = {}
    for net_id in range(args.n_parties):
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        data_loader_dict[net_id] = {}
        train_dl_local, test_dl_local, _, _ ,_,_ = get_divided_dataloader(args, dataidxs_train, dataidxs_test,traindata_cls_counts=traindata_cls_counts[net_id])
        num_classes = 100
        data_loader_dict[net_id]['train_dl_local'] = train_dl_local
        data_loader_dict[net_id]['test_dl_local'] = test_dl_local

device = args.device
config = CONFIGS[args.model_type]

### model init
net = VisionTransformerForward(config, args.img_size, num_classes=num_classes,vis = True,args= args)
net.load_from(np.load(args.pretrained_dir))
net.freeze()
net.to(device)

### params extract
global_para = {k:v.detach().clone() for k, v in net.named_parameters() if v.requires_grad == True}
# for k in global_para.keys():
#     print(k)
clients_para = {}  # client_para = {0:{k:v, k:v, ...}, 1:{k:v, k:v, ...}, ...} 
for net_id in range(args.n_parties):
    clients_para[net_id] = None

### loss save
dict_loss = {}
results_dict = defaultdict(list)

### grad
global_grad = None
client_first_grad = None

start_time = time.time()

client_arr = np.arange(args.n_parties)

### start fed_learning

for round in range(args.comm_round):
    print('########### Now is the round {} ######'.format(round))

    np.random.shuffle(client_arr)
    selected = client_arr[:args.sample_num]
    fed_avg_freqs={}

    for client_id in selected:
        print('Now is the client {}'.format(client_id))

        net.load_state_dict(global_para,strict=False)

        train_dl_local = data_loader_dict[client_id]['train_dl_local']
        param_dict = {}
        param_dict['train_dataloader'] = train_dl_local
        param_dict['round'] = round
        
        # train!
        if args.fwdtrain:
            if args.var_control:
                net = train_local_fwd_var(net, args, param_dict, client_first_grad)
            else:
                net = train_local_fwd(net, args, param_dict, client_first_grad)
            clients_para[client_id] = {k: v.clone() for k, v in net.named_parameters() if v.requires_grad == True}
        else:
            net = train_local_bp(net, args, param_dict)
            clients_para[client_id] = {k: v.clone() for k, v in net.named_parameters() if v.requires_grad == True}

    total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
    fed_avg_freqs = {r:len(net_dataidx_map_train[r]) / total_data_points for r in selected}

    client_totaltime = time.time()
    print(f'>> round {round} client total training time custom:{client_totaltime - start_time}') 
    global_para = fedavg_aggregation(clients_para, global_para, fed_avg_freqs, args, selected)

    # net.load_state_dict(global_para,strict = False) 
    
    # evaluate一下
    end_time = time.time()
    print(f'>> round {round} total time custom:{end_time - start_time}') 
    test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc = compute_accuracy_fwd(net, data_loader_dict, args)
    print('>> Mean Local Test accuracy: %f' % local_mean_acc)
    print('>> Min Local Test accuracy: %f' % local_min_acc)
    print('>> Global Model Test accuracy: %f' % test_avg_acc)
    print('>> Test avg loss: %f' %test_avg_loss)

    results_dict['test_avg_loss'].append(test_avg_loss)
    results_dict['test_avg_acc'].append(test_avg_acc)
    results_dict['local_mean_acc'].append(local_mean_acc)
    results_dict['local_min_acc'].append(local_min_acc)
    
    # for save
    # if (round+1)>=args.test_round:
    #     outfile_vit = os.path.join(save_path, 'Vit_1500_round{}.tar'.format(round))
    #     torch.save({'epoch':args.comm_round+1, 'state':net.state_dict()}, outfile_vit)

json_file_opt = "results.json"
with open(os.path.join(save_path,json_file_opt), "w") as file:
    json.dump(results_dict, file, indent=4)