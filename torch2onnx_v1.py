import os
import sys
import argparse

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from data.custom_transforms import *
from lib import *
from utils.misc import *
from data.dataloader import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--source', '-s', type=str)
    parser.add_argument('--dest', '-d', type=str, default=None)
    parser.add_argument('--type', '-t', type=str, default='map')
    parser.add_argument('--gpu', '-g', action='store_true', default=False)
    parser.add_argument('--jit', '-j', action='store_true', default=False)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    return parser.parse_args()


def get_format(source):
    img_count = len([i for i in source if i.lower().endswith(('.jpg', '.png', '.jpeg'))])
    vid_count = len([i for i in source if i.lower().endswith(('.mp4', '.avi', '.mov'))])

    if img_count * vid_count != 0:
        return ''
    elif img_count != 0:
        return 'Image'
    elif vid_count != 0:
        return 'Video'
    else:
        return ''


def torch2onnx(opt, args):
    model = eval(opt.Model.name)(**opt.Model)
    model.load_state_dict(torch.load(os.path.join(
        opt.Test.Checkpoint.checkpoint_dir, 'latest12.pth'), map_location=torch.device('cpu')), strict=True)
    model.eval()

    if os.path.isfile(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'jit.pt')) is False:
        model = Simplify(model)
        model = torch.jit.trace(model, torch.rand(1, 3, *opt.Test.Dataset.transforms.dynamic_resize.L).cuda(),
                                strict=False)
        torch.jit.save(model, os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'jit.pt'))

    else:
        del model
        model = torch.jit.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'jit.pt'))

    save_dir = None
    _format = None

    if args.source.isnumeric() is True:
        _format = 'Webcam'

    elif os.path.isdir(args.source):
        save_dir = os.path.join('results', args.source.split(os.sep)[-1])
        _format = get_format(os.listdir(args.source))

    elif os.path.isfile(args.source):
        save_dir = 'results'
        _format = get_format([args.source])

    if args.dest is not None:
        save_dir = args.dest

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)


    sample_list = eval(_format + 'Loader')(args.source, opt.Test.Dataset.transforms)
    samples = sample_list

    for sample in samples:
        with torch.no_grad():
            torch_out = model(sample)

    torch.onnx.export(model,  # 실행될 모델
                      sample,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      "InSPyReNet.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=13,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=['output'],  # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                    'output': {0: 'batch_size'}}
                      )



if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    torch2onnx(opt, args)