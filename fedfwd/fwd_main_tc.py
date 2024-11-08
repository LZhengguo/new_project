
import numpy as np
import json
import torch
from collections import  defaultdict
from pathlib import Path
import os
import copy
from math import *
import random
import numpy as np
from office_dataset import prepare_data
from domainnet_dataset import prepare_data_domain
from utils import *
import argparse
from models.vit_models_fwd import *

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='ViT-B_16', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-labeluni', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='SGPT',  help='strategy')
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
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio for each communication round')
    parser.add_argument('--test_round', type=int, default=50)  
    parser.add_argument('--root_path', type=str, default='', help='Noise type: None/increasng/space')
    parser.add_argument('--check_layer_id', type=int, default=49)
    parser.add_argument('--var_threshold', type=float, default=0.5)
    parser.add_argument('--h', type=float, default=0.01, help='learning rate (default: 0.01)')
    """
    Used for model 
    """
    parser.add_argument('--model_type', type=str, default='ViT-B_16')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained_dir', type=str, default=None, help='The pretrain model path')
    parser.add_argument('--cls_num', type=int, default=10) 
    parser.add_argument('--fwdtrain_grad',action='store_true')
    parser.add_argument('--fwdtrain_param',action='store_true')
    parser.add_argument('--bptrain',action='store_true')
    parser.add_argument('--peftmode', type=str, help='set peft mode in (adapter、lora、bitfit)')
    parser.add_argument('--bpfirst',action='store_true')
    
    args = parser.parse_args()
    return args
args = get_args()
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
# args=Args()
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

if args.dataset not in ['office','domainnet']:
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
                args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir,args=args)

# arr = [0,1,2,3]
arr = np.arange(args.n_parties)

if args.dataset == 'office':
    data_loader_dict,net_dataidx_map_train = prepare_data(args)
    num_classes = 10
elif args.dataset == 'domainnet':
    data_loader_dict,net_dataidx_map_train = prepare_data_domain(args)
    num_classes = 10
else:
    ###### Data Set related ###### 
    data_loader_dict = {}
    for net_id in arr:
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        data_loader_dict[net_id] = {}
        train_dl_local, test_dl_local, _, _ ,_,_ = get_divided_dataloader(args, dataidxs_train, dataidxs_test,traindata_cls_counts=traindata_cls_counts[net_id])
        num_classes = 100
        data_loader_dict[net_id]['train_dl_local'] = train_dl_local
        data_loader_dict[net_id]['test_dl_local'] = test_dl_local
            
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

device = args.device
###### Model related ###### 
config = CONFIGS[args.model_type]

net = VisionTransformerForward(config, args.img_size, num_classes=num_classes,vis = True,args= args)
net.load_from(np.load(args.pretrained_dir))

net.freeze()
net.to(device)

global_para = {k: v.data.clone() for k, v in net.state_dict().items() if "head" in k or "adapter" in k}
net_struct = dict([(k, v.shape) for k, v in global_para.items()])
# trainable_para = {k: copy.deepcopy(v) for k, v in net.named_parameters() if v.requires_grad == True}
# all_para = {k: copy.deepcopy(v) for k, v in net.named_parameters()}
# trainable_cnt = len(trainable_para)
# all_cnt = len(all_para)
# print(trainable_cnt)
# print(all_cnt)


clients_para = {}
# client_para = {0:{k:v, k:v, ...}, 1:{k:v, k:v, ...}, ...} 
# for net_id in arr:
#     clients_para[net_id] = {k: v.data.clone() for k, v in global_para.items()}

dict_loss = {}
results_dict = defaultdict(list)

# 用于维护合格的前向梯度
global_grad = None
total_data_points = sum([len(net_dataidx_map_train[r]) for r in range(args.n_parties)])
fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in range(args.n_parties)]
client_first_grad = None

ratio = 1
learning_rate = args.lr * ratio
optimizer = optim.SGD([p for k,p in net.named_parameters() if p.requires_grad  and  ('head' in k or 'adapter' in  k )], lr=learning_rate)

# 开始fed_learning
if args.bptrain or args.fwdtrain_grad:
    for round in range(args.comm_round):
        print('########### Now is the round {} ######'.format(round))
        
        # 维护每轮的前向梯度
        fwdgrad_pool = {i:[] for i in range(args.n_parties)}

        for client_id in range(args.n_parties):
            param_dict = {}
            # print('Now is the client {}'.format(client_id))
            
            # 每轮每个client开始先接收global参数
            # clients_para[client_id] = {k: v.data.clone() for k, v in global_para.items()}
            # net.load_state_dict(global_para,strict = False)
            
            # data_loader_dict = {0:{'train_dl_local':amazon_train_loader}, 1:{'train_dl_local':caltech_train_loader}, ...}
            train_dl_local = data_loader_dict[client_id]['train_dl_local']
            # print(len(train_dl_local),len(train_dl_local.dataset),len(train_dl_local.dataset.dataset))
            # test_dl_local = data_loader_dict[client_id]['test_dl_local'] 

            param_dict['train_dataloader'] = train_dl_local
            param_dict['round'] = round
            # train!
            if args.fwdtrain_grad:
                # 要不试下首轮用真梯度给方向 不妨先在client上算 验证想法
                if round==0 and args.bpfirst:
                    client_first_grad = use_bp_first(net, args, param_dict)

                param_dict['old_grad'] = global_grad

                fwdgrad_pool[client_id] = train_local_fwd(net, args, param_dict, client_first_grad)
            else:
                net = train_local_bp(net, args, param_dict)
                # 保存client参数
                clients_para[client_id] = {k: copy.deepcopy(v) for k, v in net.named_parameters() if v.requires_grad == True}
            

        if args.fwdtrain_grad:
            global_grad = fedsgd_aggregation_fwd_fwdgrad(net, optimizer, net_struct, fed_avg_freqs, args, fwdgrad_pool)
            # global_para = {k: v.data.clone() for k, v in net.state_dict().items() if "head" in k or "adapter" in k}
        else:
            global_para = fedavg_aggregation_bp(clients_para, global_para, fed_avg_freqs, args)
            net.load_state_dict(global_para,strict = False)
            # global_para = {k:copy.deepcopy(v.detach()) for k, v in avg_global_para.items()}

        # net.load_state_dict(global_para,strict = False)
    
        # evaluate一下
        test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc = compute_accuracy_fwd(net, data_loader_dict, args)
        print('>> Mean Local Test accuracy: %f' % local_mean_acc)
        print('>> Min Local Test accuracy: %f' % local_min_acc)
        print('>> Global Model Test accuracy: %f' % test_avg_acc)
        print('>> Test avg loss: %f' %test_avg_loss)

        results_dict['test_avg_loss'].append(test_avg_loss)
        results_dict['test_avg_acc'].append(test_avg_acc)
        results_dict['local_mean_acc'].append(local_mean_acc)
        results_dict['local_min_acc'].append(local_min_acc)
        
        # if (round+1)>=args.test_round:
        #     outfile_vit = os.path.join(save_path, 'Vit_1500_round{}.tar'.format(round))
        #     torch.save({'epoch':args.comm_round+1, 'state':net.state_dict()}, outfile_vit)
elif args.fwdtrain_param:
    for round in range(args.comm_round):
        print('########### Now is the round {} ######'.format(round))
        
        # 维护每轮的前向梯度
        fwdgrad_pool = {i:[] for i in range(args.n_parties)}

        for client_id in range(args.n_parties):
            param_dict = {}
            print('Now is the client {}'.format(client_id))
            
            # 每轮每个client开始先接收global参数
            clients_para[client_id] = {k:copy.deepcopy(v) for k, v in global_para.items()}
            net.load_state_dict(clients_para[client_id],strict = False)
            
            # data_loader_dict = {0:{'train_dl_local':amazon_train_loader}, 1:{'train_dl_local':caltech_train_loader}, ...}
            train_dl_local = data_loader_dict[client_id]['train_dl_local']
            # test_dl_local = data_loader_dict[client_id]['test_dl_local'] 

            param_dict['train_dataloader'] = train_dl_local
            param_dict['round'] = round
            # train!

            # 要不试下首轮用真梯度给方向 不妨先在client上算 验证想法
            if round==0 and args.bpfirst:
                client_first_grad = use_bp_first(net, args, param_dict)

            fwdgrad_list = train_local_fwd_(net, args, param_dict, client_first_grad)
            # train完更新下fwdgradpool的参数
            for fwdgrad in fwdgrad_list:
                fwdgrad_pool[client_id].append(fwdgrad) 

            
        global_para = fedsgd_aggregation_fwd_param(net, clients_para, fed_avg_freqs, args, fwdgrad_pool)
        # global_para = {k: copy.deepcopy(v) for k, v in net.state_dict().items() if "head" in k or "adapter" in k}
        # global_para = {k:copy.deepcopy(v.detach()) for k, v in avg_global_para.items()}

        # net.load_state_dict(global_para,strict = False)
    
        # evaluate一下
        test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc = compute_accuracy_fwd(net, data_loader_dict, args)
        print('>> Mean Local Test accuracy: %f' % local_mean_acc)
        print('>> Min Local Test accuracy: %f' % local_min_acc)
        print('>> Global Model Test accuracy: %f' % test_avg_acc)
        print('>> Test avg loss: %f' %test_avg_loss)

        results_dict['test_avg_loss'].append(test_avg_loss)
        results_dict['test_avg_acc'].append(test_avg_acc)
        results_dict['local_mean_acc'].append(local_mean_acc)
        results_dict['local_min_acc'].append(local_min_acc)
        
        # if (round+1)>=args.test_round:
        #     outfile_vit = os.path.join(save_path, 'Vit_1500_round{}.tar'.format(round))
        #     torch.save({'epoch':args.comm_round+1, 'state':net.state_dict()}, outfile_vit)

json_file_opt = "results.json"
with open(os.path.join(save_path,json_file_opt), "w") as file:
    json.dump(results_dict, file, indent=4)