"""Original real33 model with incremental inverse-square relation correction.

The existing coefficient/relation methods remain available for comparisons.
The default init_rendering/test calls select their spatially corrected variants.
The approved inference path uses an identity input copy and no denoiser.
"""

from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm
import torch
import numpy as np
import os.path as osp
from utils import svBRDF

from .author_real_capture_adapter import AuthorTwoBranchRealNet as TwoBranchRealNet

class NFPLightModel():

    def __init__(self, args):
        self.args = args
        self.device = torch.device(getattr(args, 'device', 'cuda'))
        self.renderer = svBRDF()
        self.init_distance()
        self.init_rendering()
        self.init_network()
    def init_distance(self):
        self.near_distance = 4
        self.far_distance = 12
    def init_rendering(self):
        self.coefficient, self.time_co_map = self.coefficientGenerationSpatialAttenuation()
        self.indenty = torch.ones(1, 1, self.args.image_size, self.args.image_size,
                                  dtype=torch.float32, device=self.device)
    def coefficientGeneration(self):
        y, x = np.ogrid[:self.args.image_size, :self.args.image_size]
        center_y, center_x = (self.args.image_size-1)/2, (self.args.image_size-1)/2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)/(self.args.image_size/2.0)
        angle1 = np.arctan(distance/self.near_distance)
        angle2 = np.arctan(distance/self.far_distance)
        coefficient = np.cos(angle2)-np.cos(angle1)
        coefficient = coefficient / np.cos(angle1)
        coefficient_max = np.max(coefficient)
        coefficient = coefficient / coefficient_max
        torch_coefficient = torch.Tensor(coefficient,device="cpu").to(self.device)
        torch_coefficient = torch_coefficient.unsqueeze(0).unsqueeze(0)

        time_coeff = np.cos(angle1) / np.cos(angle2)
        torch_time_coefficient = torch.Tensor(time_coeff,device="cpu").to(self.device)
        torch_time_coefficient = torch_time_coefficient.unsqueeze(0).unsqueeze(0)
        return torch_coefficient,torch_time_coefficient

    def coefficientGenerationPaper(self):
        """Generate the paper's Eq. (5) denominator for a paired experiment.

        This intentionally remains separate from ``coefficientGeneration`` so
        the existing real-inference path keeps its historical coefficient.
        The paper coefficient is max-normalized in the same way as the legacy
        coefficient to isolate the spatial denominator shape in comparisons.
        """
        y, x = np.ogrid[:self.args.image_size, :self.args.image_size]
        center_y, center_x = (self.args.image_size-1)/2, (self.args.image_size-1)/2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)/(self.args.image_size/2.0)
        angle1 = np.arctan(distance/self.near_distance)
        angle2 = np.arctan(distance/self.far_distance)
        cos_near = np.cos(angle1)
        cos_far = np.cos(angle2)
        coefficient = cos_near * np.abs(cos_near - cos_far)
        coefficient_max = np.max(coefficient)
        self.coefficient_max_paper = coefficient_max
        coefficient = coefficient / coefficient_max
        torch_coefficient = torch.as_tensor(coefficient, dtype=torch.float32, device=self.device)
        torch_coefficient = torch_coefficient.unsqueeze(0).unsqueeze(0)

        time_coeff = cos_near / cos_far
        torch_time_coefficient = torch.as_tensor(time_coeff, dtype=torch.float32, device=self.device)
        torch_time_coefficient = torch_time_coefficient.unsqueeze(0).unsqueeze(0)
        return torch_coefficient, torch_time_coefficient
    @staticmethod
    def spatialAttenuationFactor(distance_squared_near, distance_squared_far):
        """FP32 center-normalized r_far^2/r_near^2 from the s34/s15 report.

        Distances must be positive [B,1,H,W] tensors in common world units.
        The lower central pixel matches the report's [127,127] at size 256.
        The center brightness gain is already in aligned_far and is NOT
        multiplied again here. This factor corrects distance, not normals.
        """
        if (distance_squared_near.ndim != 4
                or distance_squared_near.shape[1] != 1
                or distance_squared_near.shape != distance_squared_far.shape
                or min(distance_squared_near.shape) <= 0):
            raise ValueError("squared distances require matching [B,1,H,W] shapes")
        if distance_squared_near.device != distance_squared_far.device:
            raise ValueError("squared distances must share a device")
        for distance in (distance_squared_near, distance_squared_far):
            if distance.dtype != torch.float32:
                raise TypeError("spatial attenuation requires FP32 squared distances")
            if not torch.isfinite(distance).all() or torch.any(distance <= 0):
                raise ValueError("squared distances must be finite and positive")
        ratio = distance_squared_far / distance_squared_near
        cy, cx = [(size - 1) // 2 for size in ratio.shape[-2:]]
        factor = ratio / ratio[:, :, cy:cy+1, cx:cx+1]
        if not torch.isfinite(factor).all() or torch.any(factor <= 0):
            raise ValueError("spatial attenuation factor must be finite and positive")
        return factor

    def coefficientGenerationSpatialAttenuation(self):
        """Keep the original angular denominator; add planar inverse-square geometry.

        Use the original real model's pixel-center grid and 4/12 distances,
        rather than the synthetic report's 2.414/10 renderer geometry.
        New squared-distance and attenuation calculations are entirely FP32.
        """
        coefficient, time_coefficient = self.coefficientGeneration()
        size = self.args.image_size
        if size <= 0 or not (0 < self.near_distance < self.far_distance):
            raise ValueError("require positive size and 0 < near_distance < far_distance")
        axis = (torch.arange(size, dtype=torch.float32, device=self.device)
                - (size - 1) / 2) / (size / 2.0)
        rho_squared = (axis[:, None].square() + axis[None, :].square())[None, None]
        self.distance_squared_near = rho_squared + self.near_distance ** 2
        self.distance_squared_far = rho_squared + self.far_distance ** 2
        self.spatial_attenuation_factor = self.spatialAttenuationFactor(
            self.distance_squared_near, self.distance_squared_far)
        return coefficient, time_coefficient

    def relationmapExtractionSpatialAttenuation(self, near_img, aligned_far_central_img):
        """Apply the spatial factor once, before the legacy relation clipping.

        Input far already contains the central brightness gain. Keep the
        original angular correction, denominator and feature normalization.
        Clipping follows the combined gain * angular * distance product.
        """
        if near_img.dtype != torch.float32 or aligned_far_central_img.dtype != torch.float32:
            raise TypeError("real spatial relation requires FP32 images")
        return self.relationmapExtraction(
            near_img, aligned_far_central_img * self.spatial_attenuation_factor)

    def init_network(self):
        self.net_g = TwoBranchRealNet()
        self.net_g = self.model_to_device(self.net_g)
        self.load_network(self.net_g,self.args.loadpath_network_g)

    def log_normalization(self,img, eps=1e-2):
        return (torch.log(img+eps)-torch.log(torch.ones((1,), dtype=img.dtype, device=img.device)*eps))/(torch.log(1+torch.ones((1,), dtype=img.dtype, device=img.device)*eps)-torch.log(torch.ones((1,), dtype=img.dtype, device=img.device)*eps))



    def feed_data(self, data):
        inputs = data['inputs']
        if inputs.dtype != torch.float32:
            raise TypeError("real model inputs must remain FP32")
        if (inputs.ndim != 4 or inputs.shape[1] != 6
                or inputs.shape[-2:] != (self.args.image_size, self.args.image_size)):
            raise ValueError("real model inputs require [B,6,image_size,image_size]")
        if not torch.isfinite(inputs).all() or inputs.amin() < 0 or inputs.amax() > 1:
            raise ValueError("real model inputs must be finite linear RGB in [0,1]")
        self.inputs = inputs.to(self.device)
        self.name = data['name']

    def maskExtract(self,near_img,far_central_img,orgInputs,threshold=0.95):
        orgNear, orgFar = torch.split(orgInputs,[3,3],dim=1)

        org_near_max,_ = torch.max(orgNear,dim=1,keepdim=True)
        org_far_max,_ = torch.max(orgFar,dim=1,keepdim=True)

        org_near_mask = org_near_max > threshold
        org_far_mask = org_far_max > threshold

        ### compute the scale and mask ###
        near_max,_ = torch.max(near_img,dim=1,keepdim=True)
        far_max,_ = torch.max(far_central_img,dim=1,keepdim=True)

        mask_near = near_max > threshold
        mask_far = far_max > threshold

        mask = mask_near | mask_far | org_far_mask | org_near_mask


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
        return self.indenty-self.ClipToOne(relation_map), self.indenty-self.ClipToOne(log_relation_map)
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            # No denoiser: retain the original real33 copy-feature slots.
            pred_denoise_img = self.inputs.clamp(0, 1)
            near_img, far_img = torch.split(pred_denoise_img,[3,3],1)
            scale = torch.mean(near_img[:,:,int(self.args.image_size/2)-10:int(self.args.image_size/2)+10,int(self.args.image_size/2)-10:int(self.args.image_size/2)+10])/torch.clip(torch.mean(far_img[:,:,int(self.args.image_size/2)-10:int(self.args.image_size/2)+10,int(self.args.image_size/2)-10:int(self.args.image_size/2)+10]),min=1e-5)
            # scale = torch.where(scale<6 or scale>12,torch.ones_like(scale)*9,scale)
            scale = scale.view(-1,1,1,1)
            aligned_far_img = far_img * scale
            relation_map,log_relation_map = self.relationmapExtractionSpatialAttenuation(near_img,aligned_far_img)
            aligned_far_img = torch.clip(aligned_far_img,0,1)
            mask = self.maskExtract(near_img,aligned_far_img,self.inputs)
            inputs = torch.cat([self.inputs,pred_denoise_img,aligned_far_img],1)
            log_inputs = self.log_normalization(inputs)
            inputs = torch.cat([inputs,log_inputs,relation_map,log_relation_map,mask],1)*2-1
            self.pred_svbrdf = self.net_g(inputs)
            self.relation_map = log_relation_map
            self.pred_denoise_img = pred_denoise_img
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
            self.feed_data(val_data)
            self.test()
            img_name = self.name[0]
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
