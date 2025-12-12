import os
import torch
from torch.utils import data as data
from utils import svBRDF, FileClient, imfrombytes,img2tensor
import torch.nn.functional as F
import cv2

class SynDataset(data.Dataset):
    def __init__(self,opt):
        super(SynDataset).__init__()
        self.opt = opt
        # self.eps = torch.ones([256,256])*-0.8
        self.file_client = None
        self.io_backend_opt = {
            'type': 'disk'
        }
        
        svbrdf_folder = opt['svbrdf_root']
        svbrdf_names = os.listdir(svbrdf_folder)
        svbrdf_names.sort()
        self.svbrdf_paths = []
        for name in svbrdf_names:
            self.svbrdf_paths.append(os.path.join(svbrdf_folder,name))
        self.svBRDF_utils = svBRDF()
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
    def __getitem__(self, index):
        img_bytes = self.file_client.get(self.svbrdf_paths[index], 'brdf')
        img = imfrombytes(img_bytes, float32=True)[:,:,::-1]
        svbrdfs = self.svBRDF_utils.get_svbrdfs(img)            

        return {'svbrdfs':svbrdfs,'name':os.path.basename(self.svbrdf_paths[index])}
    def __len__(self):
        return len(self.svbrdf_paths)

class RealDataset(data.Dataset):
    def __init__(self,opt):
        super(RealDataset).__init__()
        self.opt = opt
        
        svbrdf_folder = opt['root']
        svbrdf_names = os.listdir(svbrdf_folder)
        svbrdf_names.sort()
        self.svbrdf_paths = []
        for name in svbrdf_names:
            self.svbrdf_paths.append(os.path.join(svbrdf_folder,name))

            
    def __getitem__(self, index):
        near_img = cv2.imread(os.path.join(self.svbrdf_paths[index],'near.png'))
        far_img = cv2.imread(os.path.join(self.svbrdf_paths[index],'far.png'))
        if self.opt.get('imageSize',False):
            if near_img.shape[0] != self.opt['imageSize']:
                near_img = cv2.resize(near_img,[self.opt['imageSize'],self.opt['imageSize']],interpolation=cv2.INTER_AREA)
                far_img = cv2.resize(far_img,[self.opt['imageSize'],self.opt['imageSize']],interpolation=cv2.INTER_AREA)        
        near_inputs = img2tensor(near_img.copy(),bgr2rgb=True,float32=True,normalization=True)
        far_inputs = img2tensor(far_img.copy(),bgr2rgb=True,float32=True,normalization=True)
        
        inputs = torch.cat([near_inputs,far_inputs],0)
                
        if self.opt.get("input_gamma",False):
            inputs = inputs**2.2    
            

        return {'inputs':inputs,'name':os.path.basename(self.svbrdf_paths[index])}
    def __len__(self):
        return len(self.svbrdf_paths)

def log_normalization(img, eps=1e-2):
    return (torch.log(img+eps)-torch.log(torch.ones((1,))*eps))/(torch.log(1+torch.ones((1,))*eps)-torch.log(torch.ones((1,))*eps))