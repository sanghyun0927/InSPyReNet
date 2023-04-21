import os
import sys
import warnings
from argparse import ArgumentParser
from typing import Tuple, Dict, List

import onnxruntime

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from transparent_background import Remover
from transparent_background.utils import (
    dynamic_resize,
    normalize,
    tonumpy,
    totensor,
)

from utils.misc import parse_args, load_config

warnings.filterwarnings("ignore")

# Set up the file and repository paths
file_path = os.path.abspath(__file__)
repo_path = os.path.split(file_path)[0]
sys.path.append(repo_path)


def convert_to_onnx(opt, epoch: int, dummy_batch_size: int = 32):
    print("")
    print("convert_to_onnx")
    print(f"\tepoch: {epoch}")
    print(f"\tdummy_batch_size: {dummy_batch_size}")
    print("")
    # Define the image transformation pipeline
    transform = transforms.Compose([
        dynamic_resize(L=1280),
        tonumpy(),
        normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        totensor()
    ])

    # Create a sample image and apply the transformations
    img_np = np.ones((dummy_batch_size, 1024, 1024, 3), dtype='uint8')
    img = torch.stack(
        [transform(Image.fromarray(_img_np)) for _img_np in img_np]
    )
    dummy_input = img.to('cpu')

    # Initialize the Remover model with custom settings
    ckpt_path = os.path.join(opt.Test.Checkpoint.checkpoint_dir, f"latest{epoch}.pth")
    onnx_path = os.path.join(opt.onnx_model_root, f"InSPyReNet_XB_{epoch}_batch{dummy_batch_size}.onnx")
    remover = Remover(fast=False, jit=False, device='cpu', ckpt=ckpt_path)
    model = remover.model

    # Export the PyTorch model to ONNX file
    torch.onnx.export(
        model,  # Executable model
        dummy_input,  # Model input (tuple or multiple input values are also possible)
        onnx_path,  # onnx final path
        export_params=True,  # Whether to save the trained model weights in the model file
        opset_version=13,  # ONNX version to use when converting the model
        do_constant_folding=True,  # Whether to use constant folding for optimization
        input_names=['input'],  # Name indicating the input value of the model
        output_names=['output'],  # Name indicating the output value of the model
        dynamic_axes={
            'input': {0: 'batch_size'},  # Variable length dimensions
            'output': {0: 'batch_size'}
        }
    )

    print(f"Save result - '{onnx_path}'")
    return onnx_path


if __name__ == "__main__":
    this_parser = ArgumentParser()
    this_parser.add_argument('--dummy-batch-size', type=int, default=1)

    this_args, other_args = this_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + other_args

    args = parse_args()
    opt = load_config(args.config)
    convert_to_onnx(opt, 1, dummy_batch_size=this_args.dummy_batch_size)
