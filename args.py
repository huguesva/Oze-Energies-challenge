import argparse
import yaml
import os
import torch
import torch.nn as nn
import copy
from attrdict import AttrDict
from datetime import datetime

ACTIVATION = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
OPTIM = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
LOSS = {'MSE': nn.MSELoss}
path_config_folder = os.getcwd()+'/config/'

def make_args():
    parser = argparse.ArgumentParser()
    for key, value in default_config().items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    args_dict = vars(parser.parse_args())
    
    #if a config folder is given, default arguments are replaced by the config file's arguments
    if args_dict['config']:
        l = []
        path_config = os.path.join(path_config_folder, args_dict['config'])
        for filename in os.listdir(path_config):
            exp_dict = copy.deepcopy(args_dict)
            with open(os.path.join(path_config,filename), 'r') as stream:
                config_dict = yaml.safe_load(stream)
                for key, value in config_dict.items():
                    exp_dict[key] = value
            exp_dict['log_dir'] = build_exp_folder(exp_dict)
            l.append(exp_dict)
        return l
    else:
        args_dict['log_dir'] = build_exp_folder(args_dict)
        return [args_dict]
    
def build_exp_folder(args_dict):
    path_log = os.path.join(args_dict['log_dir'],name_exp(args_dict['exp_name'], args_dict['seed']))
    os.mkdir(path_log)
    os.mkdir(os.path.join(path_log,'visu'))
    torch.save(args_dict, path_log+'/params.pt')
    return path_log

def name_exp(name, seed):
    now = datetime.now()
    s = now.strftime('%d-%m-%Y_%H-%M-%S_')
    s += str(seed) + '_'
    s += name
    return s

def default_config():
    config = AttrDict()
    # General.
    config.seed = 100
    config.exp_name = ''
    config.batch_dir = 'batches' #'/pasteur/homes/hvanhass/structured-temporal-convolution/batches'
    config.data_dir = os.getcwd()
    config.log_dir = os.getcwd()+'/runs/'
    config.config = ''
    config.validation = True
    config.num_workers = 0
    # Data.
    config.len_traj = 50
    config.recompute_label_stats = False
    # Model.
    config.net_type = '1d'
    config.init = 'kaiming'
    config.activation = 'relu'
    config.depth = [300,300,300]
    config.kernel = [3,3,3]
    config.bias = False
    config.layers = 4
    config.dim_reduc = 'PCA'
    # Training.
    config.optim_iter = 2
    config.lr = 0.0005
    config.loss = 'MSE'
    config.grad_clip = 100.0
    config.optimizer = 'adam'
    config.scheduler = None
    return config