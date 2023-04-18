import os
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime
from typing import Tuple, List

import onnxruntime

import numpy as np
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")


def rembg_normalize(
        img: Image.Image,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
) -> np.ndarray:
    im = img.convert("RGB").resize(size, Image.LANCZOS)

    im_ary = np.array(im)
    im_ary = im_ary / np.max(im_ary)

    _tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
    _tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
    _tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
    _tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

    _tmpImg = _tmpImg.transpose((2, 0, 1))

    return np.expand_dims(_tmpImg, 0).astype(np.float32)


def rembg_batch_normalize(
        images: List[Image.Image],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
) -> np.ndarray:
    _tmp_images = []

    for img in images:
        _tmp_img = rembg_normalize(img, mean, std, size)
        _tmp_images.append(_tmp_img)

    return np.concatenate(_tmp_images)


def make_mask_image(ort_out: np.ndarray, image_size: Tuple[int, int]):
    pred = ort_out[:, :, :]

    ma = np.max(pred)
    mi = np.min(pred)

    pred = (pred - mi) / (ma - mi)
    pred = np.squeeze(pred)

    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
    mask = mask.resize(image_size, Image.LANCZOS)

    return mask


def assert_all_close(test_image_path, onnx_path, dummy_batch_size):
    test_image_path = os.path.join(os.path.dirname(__file__), test_image_path)
    test_image = ImageOps.exif_transpose(Image.open(test_image_path))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = (1024, 1024)

    feed = rembg_normalize(test_image, mean, std, size)
    batch_feed = rembg_batch_normalize([test_image] * dummy_batch_size, mean, std, size)

    assert feed.shape[0] == 1
    assert feed.shape[1:] == (3, 1024, 1024)

    assert batch_feed.shape[0] == dummy_batch_size
    assert batch_feed.shape[1:] == (3, 1024, 1024)

    st = datetime.now()
    ort_session = onnxruntime.InferenceSession(onnx_path)
    print(f"ort session load time : {datetime.now() - st}")

    session_input_name = ort_session.get_inputs()[0].name
    st = datetime.now()
    ort_outs = ort_session.run(None, {session_input_name: feed})
    print(f"single prediction time: {datetime.now() - st}")

    st = datetime.now()
    batch_ort_outs = ort_session.run(None, {session_input_name: batch_feed})
    print(f"batch({dummy_batch_size}) prediction time: {datetime.now() - st}")

    print("make base mask")
    base_mask = make_mask_image(ort_outs[0][0, :, :, :], test_image.size)
    base_mask.save('mask_single.png')

    print("make batch mask")
    for i in range(0, dummy_batch_size):
        batch_mask = make_mask_image(batch_ort_outs[0][i, :, :, :], test_image.size)
        batch_mask.save(f'mask_batch_{i}.png')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('onnx_path', type=str)
    parser.add_argument('test_image_path', type=str)
    parser.add_argument('--batch-size', type=int, default=2)

    # python onnx_test.py <file.onnx> --test-batch-size 2
    args = parser.parse_args([
        '../onnx/InSPyReNet_XB_10.onnx'
    ])

    assert_all_close(args.onnx_path, args.test_image_path, args.batch_size)
