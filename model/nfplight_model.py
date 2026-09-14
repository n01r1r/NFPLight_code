import os
import math
import json
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm
import torch
import numpy as np
import os.path as osp
from torch.cuda.amp import GradScaler as GradScaler
from utils import svBRDF,imwrite,tensor2img,torch_norm
from torch.autograd import Variable

from network.nfplight_net import TwoBranchNet

FEATURE_VERSIONS = ('raw_calibrated_v2', 'sample_v1', 'legacy_batch_v0')


def checkpoint_feature_version(checkpoint):
    version = checkpoint.get('feature_version', 'legacy_batch_v0')
    if version not in FEATURE_VERSIONS:
        raise ValueError(f'unsupported feature version: {version}')
    return version

class NFPLightModel():

    def __init__(self, args):
        self.args = args
        self.feature_version = getattr(args, 'feature_version', 'sample_v1')
        if self.feature_version not in FEATURE_VERSIONS:
            raise ValueError(f'unsupported feature version: {self.feature_version}')
        requested_device = str(getattr(args, 'device', 'auto')).lower()
        if requested_device == 'auto':
            requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested but is unavailable; use --device cpu or auto'
            )
        if requested_device not in {'cpu', 'cuda'} and not requested_device.startswith('cuda:'):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
        self.device = torch.device(requested_device)
        self.renderer = svBRDF()
        self.init_distance()
        self.init_rendering()
        self.init_network()
    def init_distance(self):
        self.near_distance = 2.414
        self.far_distance = 10
    def init_rendering(self):
        # init lighting direction
        self.surface = self.renderer.surface(256,1).to(self.device)
        near_pos = np.array([0,0, self.near_distance])
        self.near_pos = torch.as_tensor(
            near_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.near_light_dir, _, self.near_light_dis, _ = self.renderer.torch_generate(self.near_pos, self.near_pos,pos=self.surface)

        far_certral_pos = np.array([0,0,self.far_distance])
        self.far_central = torch.as_tensor(
            far_certral_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.far_light_dir, _, self.far_light_dis, _ = self.renderer.torch_generate(self.far_central, self.far_central,pos=self.surface)
                
        self.coefficient,self.time_co_map = self.coefficientGeneration()
        self.indenty = torch.ones(1, 1, 256, 256, device=self.device)
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
        torch_coefficient = torch.as_tensor(
            coefficient, dtype=torch.float32, device=self.device
        )
        torch_coefficient = torch_coefficient.unsqueeze(0).unsqueeze(0)
        
        time_coeff = np.cos(angle1) / np.cos(angle2)
        torch_time_coefficient = torch.as_tensor(
            time_coeff, dtype=torch.float32, device=self.device
        )
        torch_time_coefficient = torch_time_coefficient.unsqueeze(0).unsqueeze(0)
        return torch_coefficient,torch_time_coefficient
    def init_network(self):
        self.net_g = TwoBranchNet()
        self.net_g = self.model_to_device(self.net_g)
        self.load_network(self.net_g,self.args.loadpath_network_g)


    def log_normalization(self,img, eps=1e-2):
        return (torch.log(img + eps) - math.log(eps)) / (math.log1p(eps) - math.log(eps))
    
    def render_input_images(self,svbrdf,toLDR=True):
        ##### generate near images [batch,3,256,256] #####
        near_img = self.renderer._render(svbrdf,self.near_light_dir.unsqueeze(1),self.near_light_dir.unsqueeze(1),self.near_light_dis.unsqueeze(1)).squeeze(1)
        
        ##### generate far_central imagse [batch,3,256,256] #####        
        far_img = self.renderer._render(svbrdf,self.far_light_dir.unsqueeze(1),self.far_light_dir.unsqueeze(1),self.far_light_dis.unsqueeze(1)).squeeze(1)
        
        rendered = torch.cat([near_img,far_img],1)
        if not torch.isfinite(rendered).all() or (rendered < 0).any():
            raise FloatingPointError('renderer observations must be finite nonnegative linear RGB')
        inputs = rendered.clamp(0, 1)
        if toLDR:
            inputs = torch.clip(inputs,0,1)**(1/2.2)
            inputs = inputs.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
            inputs = (inputs/255.0).float()**(2.2)
        return inputs
    
    
    def feed_data(self, data, random=True):
        svbrdf = data['svbrdfs'].to(self.device)
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

    def ClipToOne(self,feature, valid_mask=None):
        if getattr(self, 'feature_version', 'sample_v1') == 'legacy_batch_v0':
            low, high = feature.amin(), feature.amax()
        elif valid_mask is not None:
            low = feature.masked_fill(~valid_mask, float('inf')).amin(dim=(1, 2, 3), keepdim=True)
            high = feature.masked_fill(~valid_mask, -float('inf')).amax(dim=(1, 2, 3), keepdim=True)
        else:
            low = feature.amin(dim=(1, 2, 3), keepdim=True)
            high = feature.amax(dim=(1, 2, 3), keepdim=True)
        return (feature - low) / (high - low).clamp(min=1e-5)
    
    def relationmapExtraction(self,near_img,aligned_far_central_img, valid_mask=None):
        diffFeature = torch.clip(self.time_co_map*aligned_far_central_img,0,1)-near_img
        diffFeature = torch.mean(torch.abs(diffFeature), dim=1, keepdim=True)
        relation_map = diffFeature / torch.clamp(self.coefficient,min=1e-5)
        log_relation_map = torch.log(relation_map.clamp(min=1e-5))
        # relation_map = relation_map / torch.max(relation_map)
        # relation_map = torch.ones_like(relation_map)-relation_map
        return (self.indenty-self.ClipToOne(relation_map, valid_mask),
                self.indenty-self.ClipToOne(log_relation_map, valid_mask))
    def build_features(self, inputs, valid_mask=None):
        """Linear [B,6,256,256] -> 21 features, optional bool pixel support.

        Invalid source/warp pixels are blacked out and excluded from relation
        normalization, then flagged in the existing mask channel. No new channels.
        """
        if inputs.ndim != 4 or inputs.shape[0] == 0 or tuple(inputs.shape[1:]) != (6, 256, 256):
            raise ValueError(f'expected [B,6,256,256], got {tuple(inputs.shape)}')
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all() or inputs.amin() < 0 or inputs.amax() > 1:
            raise ValueError('near/far inputs must be finite linear RGB in [0,1]')
        if valid_mask is not None:
            if (valid_mask.dtype != torch.bool or valid_mask.device != inputs.device
                    or valid_mask.shape != (len(inputs), 1, 256, 256)):
                raise ValueError('valid_mask must be bool [B,1,256,256] on the input device')
            if not valid_mask.flatten(1).any(1).all():
                raise ValueError('feature construction needs valid pixels in every crop')
            inputs = inputs * valid_mask
        near, far = inputs.float().chunk(2, dim=1)
        if getattr(self, 'feature_version', 'sample_v1') == 'raw_calibrated_v2':
            # Equal calibrated lamp flux/exposure. Avoid division by dark or
            # saturated texture pixels; RAW captures must satisfy this contract.
            scale = near.new_full((len(near),), (self.far_distance / self.near_distance) ** 2)
        else:
            scale = near[:, :, 127, 127].mean(1) / far[:, :, 127, 127].mean(1).clamp(min=1e-5)
        aligned_far = far * scale[:, None, None, None]
        relation, log_relation = self.relationmapExtraction(near, aligned_far, valid_mask)
        aligned_far = aligned_far.clamp(0, 1)
        mask = self.maskExtract(near, aligned_far)
        appearance = torch.cat((near, far, aligned_far), dim=1)
        features = torch.cat((appearance, self.log_normalization(appearance), relation, log_relation, mask), dim=1)
        if valid_mask is not None:
            features = features * valid_mask
            log_relation = log_relation * valid_mask
        return features * 2 - 1, log_relation, aligned_far

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            inputs, self.relation_map, self.aligned_far_img = self.build_features(self.inputs)
            self.pred_svbrdf = self.net_g(inputs)

    def save_svbrdf(self,svbrdf,path,gamma=False):
        svbrdf = svbrdf.clone()
        svbrdf[:, :3] = torch.nn.functional.normalize(svbrdf[:, :3], dim=1)
        if getattr(self, 'feature_version', None) == 'raw_calibrated_v2':
            svbrdf[:, 6:7].clamp_(min=-0.9)
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

    def save_svbrdf_maps(self, svbrdf, path, metadata=None):
        """Save one prediction as authoritative float32 HWC maps + JSON.

        Normal: unit OpenGL tangent vector; diffuse/specular: linear RGB [0,1];
        roughness: effective perceptual GGX roughness [.05,1]. Input tensor is
        normalized [1,10,H,W]. PNG montages are display previews only.
        """
        if svbrdf.ndim != 4 or svbrdf.shape[0] != 1 or svbrdf.shape[1] != 10 or min(svbrdf.shape) < 1:
            raise ValueError('float map export expects [1,10,H,W]')
        if not svbrdf.is_floating_point() or not torch.isfinite(svbrdf).all() or svbrdf.amin() < -1.00001 or svbrdf.amax() > 1.00001:
            raise ValueError('float map export expects finite normalized [-1,1] prediction')
        values = svbrdf.detach().float().clamp(-1, 1)
        lengths = values[:, :3].norm(dim=1, keepdim=True)
        if (lengths < 1e-8).any():
            raise ValueError('cannot export a zero predicted normal as a unit vector')
        maps = {
            'normal': values[:, :3] / lengths,
            'diffuse': (values[:, 3:6] + 1) / 2,
            'roughness': ((values[:, 6:7] + 1) / 2).clamp(min=0.05),
            'specular': (values[:, 7:10] + 1) / 2,
        }
        path = Path(path)
        if path.suffix.lower() in ('.png', '.npz'):
            path = path.with_suffix('')
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = Path(str(path) + '.npz')
        np.savez_compressed(archive, **{
            key: value[0].permute(1, 2, 0).cpu().numpy() for key, value in maps.items()
        })
        document = {
            'format': 'nfplight.svbrdf.linear.v1',
            'layout': 'HWC', 'dtype': 'float32', 'working_space': 'linear_srgb',
            'normal': 'unit OpenGL tangent-space xyz; negative-z predictions are not silently flipped',
            'diffuse': 'linear RGB reflectance [0,1]',
            'specular': 'linear RGB F0 [0,1]',
            'roughness': 'perceptual GGX roughness [.05,1]; alpha=roughness**2',
            'feature_version': getattr(self, 'feature_version', 'legacy_batch_v0'),
            'negative_normal_z_fraction': float((maps['normal'][:, 2] < 0).float().mean()),
            'roughness_floor_fraction': float(((values[:, 6:7] + 1) / 2 < 0.05).float().mean()),
            'normal_head': getattr(self, 'normal_head', 'xyz'),
            'provenance': metadata or {},
        }
        Path(str(path) + '.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return archive

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
        self.feature_version = checkpoint_feature_version(load_net)
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
