import argparse
import torch
from os import path as osp
import os
from data.dataset import SynDataset
from model.nfplight_model import NFPLightModel


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root',type=str,required=True,help="root path to save results.")
    parser.add_argument('--test_data_root',type=str,required=True,help="root path for data.")
    parser.add_argument('--loadpath_network_g',type=str,required=True)
    
    args = parser.parse_args()

    makedirs(args.save_root)


    return args


def create_dataloader(args):
      
    dataset_opt = {
        'name': 'SynDataset',
        'svbrdf_root': args.test_data_root,
        'log': True,
    }
    test_set = SynDataset(dataset_opt)
    dataloader_opt = {
        'dataset': test_set,
        'batch_size': 1,
        'shuffle': False,
        'pin_memory': True,
        'num_workers': 1,
    }
    test_loader = torch.utils.data.DataLoader(**dataloader_opt)
    return test_loader

def test_pipeline(args):
    torch.backends.cudnn.benchmark = True


    # create train and validation dataloaders
    test_loader = create_dataloader(args)

    # create model
    model = NFPLightModel(args)

    model.validation(test_loader)
    
        

if __name__ == '__main__':
    args = parse_options()
    
    test_pipeline(args)
