#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import torch

class svBRDF():
    def __init__(self):
        self.lampIntensity = 16

    def surface(self, size,range=1, dtype=torch.float32, device=None):
        x_range = torch.linspace(-range,range,size, dtype=dtype, device=device)
        y_range = torch.linspace(-range,range,size, dtype=dtype, device=device)
        y_mat, x_mat = torch.meshgrid(x_range, y_range)
        pos = torch.stack([x_mat, -y_mat, torch.zeros(x_mat.shape, dtype=dtype, device=device)],axis=0)
        pos = torch.unsqueeze(pos,0)
        return pos

    def torch_generate(self, camera_pos_world, light_pos_world,surface=None, pos = None):
        size = 256
        nl,_ = light_pos_world.shape
        nv,_ = camera_pos_world.shape
        if pos is None and surface is None:
            pos = self.surface(size)
        elif surface is not None:
            pos = surface

        light_pos_world = light_pos_world.reshape(nl,1,1,3)
        camera_pos_world = camera_pos_world.reshape(nv,1,1,3)

        light_pos_world = light_pos_world.permute(0,3,1,2).contiguous()
        camera_pos_world = camera_pos_world.permute(0,3,1,2).contiguous()
        view_dir_world = torch_norm(camera_pos_world - pos, dim=1)

        # pos = torch.tile(pos,[n,1,1,1])
        light_dis_square = torch.sum(torch.square(light_pos_world - pos),1,keepdims=True)

        light_dir_world = torch_norm(light_pos_world - pos, dim=1)

        return light_dir_world, view_dir_world, light_dis_square, pos

    def _render(self,inputs,l,v,dis):
        INV_PI = 1.0 / math.pi
        eps = torch.ones_like(dis)*1e-12
        def GGX(NoH, roughness):
            alpha = roughness  * roughness
            tmp = alpha / torch.max(eps,  (NoH * NoH * (alpha * alpha - 1.0) + 1.0  ) )
            return tmp * tmp * INV_PI

        def SmithG(NoV, NoL, roughness):
            def _G1(NoM, k):
                return 1 / (NoM * (1.0 - k ) + k)

            k = torch.max(eps, roughness * roughness * 0.5)
            return _G1(NoL,k) * _G1(NoV, k)

        def Fresnel(F0, VoH):
            coeff = VoH * (-5.55473 * VoH - 6.98316)
            return F0 + (1.0 - F0) *(2**coeff)

        n,d,r,s = self._seperate_brdf(inputs)

        r = r*0.5+0.5
        d = d*0.5+0.5
        s = s*0.5+0.5
        dim = -3
        n = torch_norm(n, dim=dim)
        h = torch_norm((l+v) * 0.5 , dim=dim)

        r = torch.max(r, torch.ones_like(r)*0.05)
        NoH = torch_dot(n,h, dim=dim)
        NoV = torch_dot(n,v, dim=dim)
        NoL = torch_dot(n,l, dim=dim)
        VoH = torch_dot(v,h, dim=dim)

        NoH = torch.max(NoH,eps )
        NoV = torch.max(NoV, eps)
        NoL = torch.max(NoL,eps)
        VoH = torch.max(VoH, eps)

        f_d = d  * INV_PI

        D = GGX(NoH,r)
        G = SmithG(NoV, NoL, r)
        F = Fresnel(s, VoH)
        f_s = D * G * F * 0.25

        res = (f_d + f_s) * NoL /dis
        res *= self.lampIntensity
        return res

    def _seperate_brdf(self, svbrdf):
        if svbrdf.ndim != 4 or svbrdf.shape[1] != 10:
            raise ValueError("renderer requires [B,10,H,W] normalized n/d/r/s")
        return tuple(svbrdf[:, channels].unsqueeze(1) for channels in
                     (slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 10)))

def torch_norm(arr, dim=1):
    length = torch.sqrt(torch.sum(arr * arr, dim = dim, keepdims=True))
    return arr / (length + 1e-12)

def torch_dot(a,b, dim=-3):
    return torch.sum(a*b,dim=dim,keepdims=True)

