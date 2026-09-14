import argparse
import torch
from os import path as osp
import os
from data.dataset import SynDataset
from model.nfplight_model import NFPLightModel
from model.nfplight_model import checkpoint_feature_version
from model.normal_head import checkpoint_normal_head


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def validate_synthetic_checkpoint(loadpath):
    """Reject Fabric-only contracts before entering the synthetic path."""
    checkpoint = torch.load(loadpath, map_location='cpu', weights_only=False)
    feature_version = checkpoint_feature_version(checkpoint)
    normal_head = checkpoint_normal_head(checkpoint)
    if feature_version == 'raw_calibrated_v2':
        raise ValueError(
            'raw_calibrated_v2 Fabric checkpoints require real.py '
            '--estimator-family matsynth21 with linear capture inputs'
        )
    if normal_head != 'xyz':
        raise ValueError(
            f'normal head {normal_head!r} is not supported by test.py; '
            'use real.py --estimator-family matsynth21'
        )
    return feature_version, normal_head
    
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

    validate_synthetic_checkpoint(args.loadpath_network_g)


    # create train and validation dataloaders
    test_loader = create_dataloader(args)

    # create model
    model = NFPLightModel(args)

    model.validation(test_loader)
    
        

if __name__ == '__main__':
    args = parse_options()
    
    test_pipeline(args)
