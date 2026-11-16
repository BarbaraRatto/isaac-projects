"""CPU executable Torch parity tests; also exercise CUDA whenever available."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
import torch
from train_general_big_map_test7_SAC.check_gpu_maps import check_maps, check_backprojection

DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
def test_torch_maps_match_cpu(device):
    check_maps(device)

@pytest.mark.parametrize('device', DEVICES)
def test_torch_backprojection_matches_cpu(device):
    check_backprojection(device)

@pytest.mark.parametrize('device', DEVICES)
def test_vision_dispatch_and_features_stay_on_device(monkeypatch, device):
    import numpy as np
    from types import SimpleNamespace
    from train_general_big_map_test7_SAC import vision
    from train_general_big_map_test7_SAC.gpu_memory import TorchMapBank

    class Extractor:
        def __init__(self, source, device, size):
            self.device = torch.device(device)
            self.feature_dim = 384
        def processor(self, images, **kwargs):
            return {'pixel_values': torch.tensor(np.stack(images)).permute(0,3,1,2).float()/255}
        def model(self, pixel_values):
            tokens = torch.arange(1370*384, device=self.device, dtype=torch.float32).reshape(1,1370,384)*1e-5
            bias = pixel_values.mean((1,2,3))[:,None,None]
            return SimpleNamespace(last_hidden_state=tokens+bias)

    monkeypatch.setattr(vision, 'DinoFeatureExtractor', Extractor)
    pca = SimpleNamespace(mean=np.zeros(384,dtype=np.float32),components=np.eye(384,dtype=np.float32)[:32])
    reference = vision.LocalVision('cpu',pca,batch_size=1)
    candidate = vision.LocalVision(device,pca,batch_size=2)
    candidate.gpu_maps = True
    candidate.memory = TorchMapBank(device)
    n=2
    positions=torch.tensor([[0.,0.,1.5],[8000.,16000.,1.5]],dtype=torch.float64)
    bases=positions-torch.tensor([0.,0.,1.])
    rng=np.random.default_rng(88)
    rgb=torch.from_numpy(rng.integers(0,256,(n,360,640,4),dtype=np.uint8))
    depth=torch.full((n,360,640,1),1.5)
    data=dict(output={'rgb':rgb,'distance_to_image_plane':depth},
              intrinsic_matrices=torch.tensor([[[400.,0,320],[0,400,180],[0,0,1]]]*n),
              pos_w=positions,quat_w_ros=torch.tensor([[0.,1,0,0]]*n))
    reference_camera=SimpleNamespace(data=SimpleNamespace(**data))
    gpu_data={k:({a:b.to(device) for a,b in v.items()} if isinstance(v,dict) else v.to(device)) for k,v in data.items()}
    candidate_camera=SimpleNamespace(data=SimpleNamespace(**gpu_data))
    for step in range(3):
        yaw=torch.tensor([.37*step,-.21*step],dtype=torch.float64)
        moved=bases+torch.tensor([.13*step,.07*step,0.])
        bodies=bases[:,None,:]
        times=torch.tensor([.2*step,.2*step],dtype=torch.float64)
        expected=reference.extract(reference_camera,moved,yaw,bodies,times)
        actual=candidate.extract(candidate_camera,*[x.to(device) for x in (moved,yaw,bodies,times)])
        assert actual.device==torch.device(device)
        torch.testing.assert_close(actual.cpu(),expected,rtol=2e-5,atol=2e-6)
        if step==1:
            reference.reset(torch.tensor([1]));candidate.reset(torch.tensor([1],device=device))

@pytest.mark.parametrize('device', DEVICES)
def test_sparse_bins_overwrite_and_degenerate_geometry(device):
    from train_general_big_map_test7_SAC.gpu_memory import TorchMapBank
    bank=TorchMapBank(device)
    t=lambda x,**kw:torch.tensor(x,device=device,**kw)
    base=t([[0.,0.,.5]],dtype=torch.float64)
    patch=t([[.5,0.,0.]],dtype=torch.float64)
    env=t([0],dtype=torch.long)
    for value in (1.,2.):
        out=bank.observe(patch,torch.full((1,32),value,device=device),env,
                         patch,env,base,t([0.]),t([value]))
        assert bank.visual.weight.sum()==1
        assert out[:,:,37].sum()==0  # one point cannot fabricate a plane
        assert out[:,:,36].sum()==1
        assert (out[0,out[0,:,36].bool(),:32]==value).all()
    with pytest.raises(ValueError,match='Non-finite'):
        bank.observe(patch,torch.full((1,32),float('nan'),device=device),env,patch,env,base,t([0.]),t([3.]))
