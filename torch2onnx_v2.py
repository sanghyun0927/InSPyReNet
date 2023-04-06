import os
import sys
import warnings
import torchvision.transforms as transforms


filepath = os.path.abspath(__file__)
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from transparent_background.utils import *
from transparent_background import Remover

warnings.filterwarnings("ignore")

CONFIG = {
'base':     {'url': "https://drive.google.com/file/d/13oBl5MTVcWER3YU4fSxW3ATlVfueFQPY/view?usp=share_link",
             'md5': "d692e3dd5fa1b9658949d452bebf1cda",
             'base_size': [1024, 1024],
             'threshold': None,
             'ckpt_name': "ckpt_base.pth",
             'resize': dynamic_resize(L=1280)},
'fast':     {'url': "https://drive.google.com/file/d/1iRX-0MVbUjvAVns5MtVdng6CQlGOIo3m/view?usp=share_link",
             'md5': "9efdbfbcc49b79ef0f7891c83d2fd52f",
             'base_size': [384, 384],
             'threshold': 512,
             'ckpt_name': "ckpt_fast.pth",
             'resize': static_resize(size=[384, 384])}
}

meta = CONFIG['base']
# ckpt = './checkpoints/ckpt_base.pth'
#
# ckpt_dir, ckpt_name = os.path.split(os.path.abspath(ckpt))
#
# model = InSPyReNet_SwinB(depth=64, pretrained=False, **meta)
# model.eval()
# model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name), map_location='cpu'))
# # model = model.to(device)

transform = transforms.Compose([meta['resize'],
                                tonumpy(),
                                normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                                totensor()])

img_np = np.ones((1024, 1024, 3), dtype='uint8')
img = Image.fromarray(img_np)  # read image
shape = img.size[::-1]
x = transform(img)
x = x.unsqueeze(0)
x = x.to('cpu')

remover = Remover(fast=False, jit=False, device='cpu', ckpt='./checkpoints/latest20.pth')  # custom setting
model = remover.model

torch_out = model(x)
print(torch_out.shape)

torch.onnx.export(model,  # 실행될 모델
                  x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "InSPyReNet_XB_20.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=13,  # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names=['input'],  # 모델의 입력값을 가리키는 이름
                  output_names=['output'],  # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                'output': {0: 'batch_size'}}
                  )
