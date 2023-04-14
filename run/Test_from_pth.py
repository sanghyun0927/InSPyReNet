import os
import sys
import tqdm

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.torch2onnx import get_ort_session
from data.dataloader import *
from cargen.utils.u2net_bg import remove


def test(opt, epoch):
    session = get_ort_session(opt, epoch)

    set = opt.Test.Dataset.sets[0]
    dataset_dir = os.path.join(opt.Test.Dataset.root, set)
    image_dir = os.path.join(dataset_dir, 'images')
    # mask_dir = os.path.join(dataset_dir, 'masks')

    for image_name in tqdm.tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        save_path = os.path.join(opt.Test.Checkpoint.checkpoint_dir, f"{set}_epoch{epoch}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image = Image.open(image_path)
        mask = remove(image, session, post_process_mask=True, only_mask=True, size=1024)

        mask.save(os.path.join(save_path, image_name))


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    for epoch in tqdm.tqdm(range(1, 10)):
        test(opt, epoch)
