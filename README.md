# NFPLight
This is the source code of research paper "NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point lighting" (SIGGRAPH Asia 2024).
**More information (include our paper, supplementary, video) can be found at** [My Personal Page](https://cgliwang.github.io/) 
![Alt](Teaser.jpg)

# Pretrained models
Our pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/171Krs3DUGqI-IkejbtOsbPtrKpOlds2p?usp=sharing). Unzip these files to 'checkpoints' folder.

# Dependencies
```python
conda create --name new_env --file requirements.txt
```
- Python (tested on 3.10)
- Pytorch (tested on 2.9.1+CUDA 12.8)

# Usage
- Test on synthetic data

This script tests the method on synthetic data under ideal capture settings, 
aiming to evaluate its theoretical upper-bound performance.
```Python
python test.py
  --save_root ./results/syn
  --test_data_root ./input_data/syn_data
  --loadpath_network_g ./checkpoints/net_g_syn.pth
```

- Test on real data

This script incorporates a denoising network for real material capture.
The model is trained at a resolution of 256, but the proposed method supports inference at 1024
to produce higher-resolution results.
```Python
python real.py
  --save_root ./results/syn
  --test_data_root ./input_data/real_data
  --loadpath_network_g ./checkpoints/net_g_real.pth
  --loadpath_network_denoise ./checkpoints/net_denoise_real.pth
  --image_size 1024
```

# Citation
If you use our code or pretrained models, please cite as following:
```
@inproceedings{10.1145/3757377.3763905,
author = {Wang, Li and Zhao, Jiajun and Zhang, Lianghao and Gao, Fangzhou and Zhang, Jiawan},
title = {EBREnv: SVBRDF Estimation in Uncontrolled Environment Lighting via Exemplar-Based Representation},
year = {2025},
isbn = {9798400721373},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3757377.3763905},
doi = {10.1145/3757377.3763905},
abstract = {Recovering spatial-varying bi-directional reflectance distribution function (SVBRDF) from as few as possible captured images has been a challenging task in computer graphics. Benefiting from the co-located flashlight-camera capture strategy and data-driven priors, SVBRDF can be estimated from few input images. However, this capture strategy usually requires a controllable darkroom environment, ensuring the flashlight is a single light source. It is often impractical during on-site capture in real-world scenarios. To support SVBRDF estimation in an uncontrolled environment, the key challenge lies in the high-precise estimation of unknown environment lighting and its effective utilization on SVBRDF recovery. To address this issue, we proposed a novel exemplar-based environment lighting representation, which is easier to use for neural networks. These exemplars are a set of rendered images of selected materials under the environment lighting. By embedding the rendering process, our approach transforms environment lighting represented in the spherical domain into the sample-surface domain, thereby achieving the domain alignment with input images. This significantly reduces the network’s learning burden, resulting in a more precise environment lighting estimation. Furthermore, after lighting prediction, we also present a dominant lighting extraction algorithm and an adaptive exemplar selection algorithm to enhance the guidance of environment lighting in SVBRDF estimation. Finally, considering the distant contribution of environment lighting and point lighting to SVBRDF recovery, we proposed a well-designed cascaded network. Quantitative assessments and qualitative analysis have demonstrated that our method achieves superior SVBRDF estimations compared to previous approaches. The source code will be released.},
booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
articleno = {163},
numpages = {10},
keywords = {Material Reflectance Modeling, SVBRDF, Deep Learning, Environment Lighting},
location = {
},
series = {SA Conference Papers '25}
}
```
