import os
from collections import OrderedDict
from copy import deepcopy
from tkinter import image_names
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm
import torch
import numpy as np
import os.path as osp
from torch.cuda.amp import GradScaler as GradScaler
from utils import svBRDF,imwrite,tensor2img,torch_norm
from torch.autograd import Variable

from network.nfplight_net import TwoBranchNet

class NFPLightModel():

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')    
        self.renderer = svBRDF()
        self.init_distance()
        self.init_rendering()
        self.init_network()
    def init_distance(self):
        self.near_distance = 2.414
        self.far_distance = 10
    def init_rendering(self):
        # init lighting direction
        self.surface = self.renderer.surface(256,1).to('cuda')
        near_pos = np.array([0,0, self.near_distance])
        self.near_pos = torch.Tensor(near_pos,device="cpu").cuda().unsqueeze(0)
        self.near_light_dir, _, self.near_light_dis, _ = self.renderer.torch_generate(self.near_pos, self.near_pos,pos=self.surface)

        far_certral_pos = np.array([0,0,self.far_distance])
        self.far_central = torch.Tensor(far_certral_pos,device="cpu").cuda().unsqueeze(0)
        self.far_light_dir, _, self.far_light_dis, _ = self.renderer.torch_generate(self.far_central, self.far_central,pos=self.surface)
                
        self.coefficient,self.time_co_map = self.coefficientGeneration()
        self.indenty = torch.ones(1, 1, 256, 256).cuda()
    def coefficientGeneration(self):
        y, x = np.ogrid[:256, :256]
        center_y, center_x = 127.5,127.5
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)/128
        angle1 = np.arctan(distance/self.near_distance)
        angle2 = np.arctan(distance/self.far_distance)
        coefficient = np.cos(angle2)-np.cos(angle1)
        coefficient = coefficient / np.cos(angle1)
        coefficient_max = np.max(coefficient)
        coefficient = coefficient / coefficient_max
        torch_coefficient = torch.Tensor(coefficient,device="cpu").cuda()
        torch_coefficient = torch_coefficient.unsqueeze(0).unsqueeze(0)
        
        time_coeff = np.cos(angle1) / np.cos(angle2)
        torch_time_coefficient = torch.Tensor(time_coeff,device="cpu").cuda()
        torch_time_coefficient = torch_time_coefficient.unsqueeze(0).unsqueeze(0)
        return torch_coefficient,torch_time_coefficient
    def init_network(self):
        self.net_g = TwoBranchNet()
        self.net_g = self.model_to_device(self.net_g)
        self.load_network(self.net_g,self.args.loadpath_network_g)


    def log_normalization(self,img, eps=1e-2):
        return (torch.log(img+eps)-torch.log(torch.ones((1,),device="cuda")*eps))/(torch.log(1+torch.ones((1,),device='cuda')*eps)-torch.log(torch.ones((1,),device='cuda')*eps))     
    
    def render_input_images(self,svbrdf,toLDR=True):
        ##### generate near images [batch,3,256,256] #####
        near_img = self.renderer._render(svbrdf,self.near_light_dir.unsqueeze(1),self.near_light_dir.unsqueeze(1),self.near_light_dis.unsqueeze(1)).squeeze(1)
        
        ##### generate far_central imagse [batch,3,256,256] #####        
        far_img = self.renderer._render(svbrdf,self.far_light_dir.unsqueeze(1),self.far_light_dir.unsqueeze(1),self.far_light_dis.unsqueeze(1)).squeeze(1)
        
        inputs = torch.clip(torch.cat([near_img,far_img],1),0,1)
        if toLDR:
            inputs = torch.clip(inputs,0,1)**(1/2.2)
            inputs = inputs.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
            inputs = (inputs/255.0).float()**(2.2)
        return inputs
    
    
    def feed_data(self, data, random=True):
        svbrdf = data['svbrdfs'].cuda()
        self.inputs = self.render_input_images(svbrdf,toLDR=False)
        self.name = data['name']

    def maskExtract(self,near_img,far_central_img):
        ### compute the scale and mask ###
        near_max,_ = torch.max(near_img,dim=1,keepdim=True)
        far_max,_ = torch.max(far_central_img,dim=1,keepdim=True)
        
        mask_near = near_max > 0.95
        mask_far = far_max > 0.95
        
        mask = mask_near | mask_far

        
        masks = mask.float()
        ivs_masks = torch.ones_like(masks) - masks
        return ivs_masks   

    def ClipToOne(self,feature):
        max_feature = torch.max(feature)
        min_feature = torch.min(feature)
        
        return (feature-min_feature)/torch.clip((max_feature-min_feature),min=1e-5)
    
    def relationmapExtraction(self,near_img,aligned_far_central_img):
        diffFeature = torch.clip(self.time_co_map*aligned_far_central_img,0,1)-near_img
        diffFeature = torch.mean(torch.abs(diffFeature), dim=1, keepdim=True)
        relation_map = diffFeature / torch.clamp(self.coefficient,min=1e-5)
        log_relation_map = torch.log(relation_map.clamp(min=1e-5))
        # relation_map = relation_map / torch.max(relation_map)
        # relation_map = torch.ones_like(relation_map)-relation_map
        return self.indenty-self.ClipToOne(relation_map), self.indenty-self.ClipToOne(log_relation_map)
    def test(self):            
        self.net_g.eval()
        with torch.no_grad():
            near_img, far_img = torch.split(self.inputs,[3,3],1)
            scale = torch.mean(near_img[:,:,127,127],1)/torch.clip(torch.mean(far_img[:,:,127,127],1),min=1e-5)
            scale = scale.view(-1,1,1,1)
            aligned_far_img = far_img * scale
            relation_map,log_relation_map = self.relationmapExtraction(near_img,aligned_far_img)
            aligned_far_img = torch.clip(aligned_far_img,0,1)
            mask = self.maskExtract(near_img,aligned_far_img)

            inputs = torch.cat([self.inputs,aligned_far_img],1)
            log_inputs = self.log_normalization(inputs)
            inputs = torch.cat([inputs,log_inputs,relation_map,log_relation_map,mask],1)*2-1
            self.pred_svbrdf = self.net_g(inputs)
            self.relation_map = log_relation_map
            self.aligned_far_img = aligned_far_img

    def save_svbrdf(self,svbrdf,path,gamma=False):
        svbrdf = svbrdf * 0.5 + 0.5
        n,d,r,s = torch.split(svbrdf,[3,3,1,3],1)
        r = torch.tile(r,[1,3,1,1])
        near_input, far_input = torch.split(self.inputs,[3,3],1)
        if gamma:
            near_input = near_input ** (1/2.2)
            far_input = far_input ** (1/2.2)
            d = d ** (1/2.2)
            s = s ** (1/2.2)
        svbrdf_vis = torch.cat([near_input,far_input,n,d,r,s],-1)
        import torchvision
        if '.png' not in path:
            path = path + '.png'
        torchvision.utils.save_image(svbrdf_vis,path)

    def validation(self, dataloader):
        pbar = tqdm(total=len(dataloader), unit='image')
        self. metric_results = 0
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data,random=False)
            self.test()
            img_name = self.name[0][:-4]
            path = osp.join(self.args.save_root, img_name)
            self.save_svbrdf(self.pred_svbrdf,path,gamma=True)

            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Testing')

        pbar.close()


    def save_visuals(self):
        svbrdf_gt_pre = torch.cat([self.svbrdf,self.fake_svbrdf],-2)
        normal,diffsue, roughness,specular = torch.split(svbrdf_gt_pre,[3,3,1,3],1)
        svbrdf_normal = torch.cat([normal,diffsue, torch.tile(roughness,(1,3,1,1)),specular],-1)

        return svbrdf_normal.squeeze(0)
    def model_to_device(self, net):
        net = net.float().to(self.device)
        return net

    def get_current_log(self):
        return self.log_dict
    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net
    def load_network(self, net, load_path, strict=True, param_key='params',IsCompile=False):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        # logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
            load_net = load_net[param_key]
        # remove unnecessary 'module.'

        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
            if k.startswith('step_counter'):
                load_net.pop(k)

        keys_list = list(load_net.keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                load_net[deal_key] = load_net[key]
                del load_net[key]
        net.load_state_dict(load_net, strict=strict)



    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f'  {v}')
            print('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)