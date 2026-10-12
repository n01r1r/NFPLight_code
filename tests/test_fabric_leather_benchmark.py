import torch
from benchmark_fabric_leather import fixed_f0, lights40, ndr_loss


def test_fixed_f0_removes_specular_gradient():
    maps=torch.randn(1,10,2,2,requires_grad=True)
    result=fixed_f0(maps)
    torch.testing.assert_close(result[:,7:],torch.full_like(result[:,7:],-.92))
    result.sum().backward()
    assert maps.grad[:,:7].eq(1).all()
    assert maps.grad[:,7:].eq(0).all()


def test_grid_has_exact_40_directions_radius_four():
    lights=lights40('cpu')
    assert lights.shape==(40,3)
    assert len(torch.unique(lights,dim=0))==40
    torch.testing.assert_close(lights.norm(dim=1),torch.full((40,),4.))
    assert lights[:,2].gt(0).all()


def test_ndr_equal_map_mean_ignores_specular():
    target=torch.zeros(1,10,2,2)
    pred=target.clone()
    pred[:,:3]=.3
    pred[:,3:6]=.6
    pred[:,6:7]=.9
    pred[:,7:]=100
    value,parts=ndr_loss(pred,target,torch.ones(1,1,2,2,dtype=torch.bool))
    torch.testing.assert_close(value,torch.tensor(.6))
    assert len(parts)==3
