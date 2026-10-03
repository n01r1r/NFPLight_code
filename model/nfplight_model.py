"""Shared FP32 features and rendering for Fabric training and capture inference."""
import math
import json
from pathlib import Path
import numpy as np
import torch
from utils import svBRDF

FEATURE_VERSIONS = ('raw_calibrated_v2', 'sample_v1', 'legacy_batch_v0')


def checkpoint_feature_version(checkpoint):
    version = checkpoint.get('feature_version', 'legacy_batch_v0')
    if version not in FEATURE_VERSIONS:
        raise ValueError(f'unsupported feature version: {version}')
    return version

class NFPLightModel():
    # Legacy callers construct small __new__ fixtures and subclasses without
    # running this initializer; keep their established FP32 contract.
    compute_dtype = torch.float32
    precision = 'float32'
    capture_precision = False

    def __init__(self, args):
        self.args = args
        requested_precision = str(getattr(args, 'precision', 'float32')).lower()
        if requested_precision in {'fp32', 'float', 'single'}:
            requested_precision = 'float32'
        if requested_precision != 'float32':
            raise ValueError("model execution supports only float32")
        self.precision = requested_precision
        self.capture_precision = bool(getattr(args, 'capture_precision', False))
        self.compute_dtype = torch.float32
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
        self.surface = self.renderer.surface(256,1, dtype=self.compute_dtype, device=self.device)
        near_pos = np.array([0,0, self.near_distance], dtype=np.float32 if self.capture_precision else None)
        self.near_pos = torch.as_tensor(
            near_pos, dtype=self.compute_dtype, device=self.device
        ).unsqueeze(0)
        self.near_light_dir, _, self.near_light_dis, _ = self.renderer.torch_generate(self.near_pos, self.near_pos,pos=self.surface)

        far_certral_pos = np.array([0,0,self.far_distance], dtype=np.float32 if self.capture_precision else None)
        self.far_central = torch.as_tensor(
            far_certral_pos, dtype=self.compute_dtype, device=self.device
        ).unsqueeze(0)
        self.far_light_dir, _, self.far_light_dis, _ = self.renderer.torch_generate(self.far_central, self.far_central,pos=self.surface)
                
        self.coefficient,self.time_co_map = self.coefficientGeneration()
        self.indenty = torch.ones(1, 1, 256, 256, dtype=self.compute_dtype, device=self.device)
    def coefficientGeneration(self):
        compute_dtype = getattr(self, 'compute_dtype', torch.float32)
        numpy_dtype = np.float64
        if self.capture_precision and compute_dtype == torch.float32:
            numpy_dtype = np.float32
        y, x = np.ogrid[:256, :256]
        x, y = x.astype(numpy_dtype), y.astype(numpy_dtype)
        center_y, center_x = numpy_dtype(127.5), numpy_dtype(127.5)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2, dtype=numpy_dtype) / numpy_dtype(128)
        angle1 = np.arctan(distance/self.near_distance)
        angle2 = np.arctan(distance/self.far_distance)
        coefficient = np.cos(angle2)-np.cos(angle1)
        coefficient = coefficient / np.cos(angle1)
        coefficient_max = np.max(coefficient)
        raw_coefficient = coefficient.copy()
        self.coefficient_raw = torch.as_tensor(
            raw_coefficient, dtype=compute_dtype, device=self.device
        ).unsqueeze(0).unsqueeze(0)
        self.coefficient_normalization_max = torch.as_tensor(
            coefficient_max, dtype=compute_dtype, device=self.device
        )
        coefficient = coefficient / coefficient_max
        torch_coefficient = torch.as_tensor(
            coefficient, dtype=compute_dtype, device=self.device
        )
        torch_coefficient = torch_coefficient.unsqueeze(0).unsqueeze(0)
        
        time_coeff = np.cos(angle1) / np.cos(angle2)
        torch_time_coefficient = torch.as_tensor(
            time_coeff, dtype=compute_dtype, device=self.device
        )
        torch_time_coefficient = torch_time_coefficient.unsqueeze(0).unsqueeze(0)
        return torch_coefficient,torch_time_coefficient


    def log_normalization(self,img, eps=1e-2):
        return (torch.log(img + eps) - math.log(eps)) / (math.log1p(eps) - math.log(eps))
    
    def render_input_images(self,svbrdf):
        ##### generate near images [batch,3,256,256] #####
        near_img = self.renderer._render(svbrdf,self.near_light_dir.unsqueeze(1),self.near_light_dir.unsqueeze(1),self.near_light_dis.unsqueeze(1)).squeeze(1)
        
        ##### generate far_central imagse [batch,3,256,256] #####        
        far_img = self.renderer._render(svbrdf,self.far_light_dir.unsqueeze(1),self.far_light_dir.unsqueeze(1),self.far_light_dis.unsqueeze(1)).squeeze(1)
        
        rendered = torch.cat([near_img,far_img],1)
        if not torch.isfinite(rendered).all() or (rendered < 0).any():
            raise FloatingPointError('renderer observations must be finite nonnegative linear RGB')
        inputs = rendered.clamp(0, 1)
        return inputs
    
    

    def maskExtract(self,near_img,far_central_img):
        ### compute the scale and mask ###
        near_max,_ = torch.max(near_img,dim=1,keepdim=True)
        far_max,_ = torch.max(far_central_img,dim=1,keepdim=True)
        
        mask_near = near_max > 0.95
        mask_far = far_max > 0.95
        
        mask = mask_near | mask_far

        
        masks = mask.to(dtype=getattr(self, 'compute_dtype', torch.float32))
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
        near, far = inputs.to(dtype=getattr(self, 'compute_dtype', torch.float32)).chunk(2, dim=1)
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






    def model_to_device(self, net):
        net = net.to(device=self.device, dtype=getattr(self, 'compute_dtype', torch.float32))
        return net

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
