import torch

def get_log_path():
    return 'G:/_tf_log_tmp/pytorch_logs/'

def get_dataset_path():
    return './../flower_data/'

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")