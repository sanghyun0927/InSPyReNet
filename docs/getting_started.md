# :flight_departure: Getting Started

## Create environment
  + Create conda environment with following command `conda create -y -n inspyrenet python`
  + Activate environment with following command `conda activate inspyrenet`
  + Install requirements with following command `pip install -r requirements.txt`
  
## Preparation

* For training, you may need training datasets and ImageNet pre-trained checkpoints for the backbone. For testing (inference), you may need test datasets (sample images).
* Training datasets are expected to be located under [Train.Dataset.root](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L10). Likewise, testing datasets should be under [Test.Dataset.root](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L58).
* Each dataset folder should contain `images` folder and `masks` folder for images and ground truth masks respectively.
* You may use multiple training datasets by listing dataset folders for [Train.Dataset.sets](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L12), such as `[DUTS-TR] -> [DUTS-TR, HRSOD-TR, UHRSD-TR]`.

### Backbone Checkpoints
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
Res2Net50 checkpoint | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUO7GDBwoC9CulTPdnq_yhQBlc0SIyyELMy3OmrNhOjcGg?e=T3PVyG) | [Link](https://drive.google.com/file/d/1MMhioAsZ-oYa5FpnTi22XBGh5HkjLX3y/view?usp=sharing)
SwinB checkpoint     | `data/backbone_ckpt/*.pth` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ESlYCLy0endMhcZm9eC2A4ABatxupp4UPh03EcqFjbtSRw?e=7y6lLt) | [Link](https://drive.google.com/file/d/1fBJFMupe5pV-Vtou-k8LTvHclWs0y1bI/view?usp=sharing)

### Train Datasets
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
DUTS-TR | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EQ7L2XS-5YFMuJGee7o7HQ8BdRSLO8utbC_zRrv-KtqQ3Q?e=bCSIeo) | [Link](https://drive.google.com/file/d/1hy5UTq65uQWFO5yzhEn9KFIbdvhduThP/view?usp=share_link)

### Extra Train Datasets (Optional)
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
HRSOD-TR | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUx92hUgZJNrWPj46PC0yEBXcorQskXOCSz8SnGH5AcLQ?e=WA5pc6) | N/A
UHRSD-TR | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ea4_UCbsKmhKnMCccAJOTLgBmQFsQ4KhJSf2jx8WQqj3Wg?e=18kYZS) | N/A
DIS-TR   | `data/Train_Dataset/...`   | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EZtZJ493tVNJjBIpNLdus68B3u906PdWtHsf87pulj78jw?e=bUg2UQ) | N/A

### Test Datasets
Item | Destination Folder | OneDrive | GDrive
:-|:-|:-|:-
DUTS-TE   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfuCxjveXphPpIska9BxHDMBHpYroEKdVlq9HsonZ4wLDw?e=Mz5giA) | [Link](https://drive.google.com/file/d/1w4pigcQe9zplMulp1rAwmp6yYXmEbmvy/view?usp=share_link) 
DUT-OMRON | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERvApm9rHH5LiR4NJoWHqDoBCneUQNextk8EjQ_Hy0bUHg?e=wTRZQb) | [Link](https://drive.google.com/file/d/1qIm_GQLLQkP6s-xDZhmp_FEAalavJDXf/view?usp=sharing) 
ECSSD     | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ES_GCdS0yblBmnRaDZ8xmKQBPU_qeECTVB9vlPUups8bnA?e=POVAlG) | [Link](https://drive.google.com/file/d/1qk_12KLGX6FPr1P_S9dQ7vXKaMqyIRoA/view?usp=sharing) 
HKU-IS    | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYBRVvC1MJRAgSfzt0zaG94BU_UWaVrvpv4tjogu4vSV6w?e=TKN7hQ) | [Link](https://drive.google.com/file/d/1H3szJYbr5_CRCzrYfhPHThTgszkKd1EU/view?usp=share_link) 
PASCAL-S  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUDGDckMnZHhEPy8YQGwBQB5MN3qInBkEygpIr7ccJdTQ?e=YarZaQ) | [Link](https://drive.google.com/file/d/1h0IE2DlUt0HHZcvzMV5FCxtZqQqh9Ztf/view?usp=sharing)
DAVIS-S   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/Ebam8I2o-tRJgADcq-r9YOkBCDyaAdWBVWyfN-xCYyAfDQ?e=Mqz8cK) | [Link](https://drive.google.com/file/d/15F0dy9o02LPTlpUbnD9NJlGeKyKU3zOz/view?usp=sharing)
HRSOD-TE  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EbHOQZKC59xIpIdrM11ulWsBHRYY1wZY2njjWCDFXvT6IA?e=wls17m) | [Link](https://drive.google.com/file/d/1KnUCsvluS4kP2HwUFVRbKU8RK_v6rv2N/view?usp=sharing)
UHRSD-TE  | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EUpc8QJffNpNpESv-vpBi40BppucqOoXm_IaK7HYJkuOog?e=JTjGmS) | [Link](https://drive.google.com/file/d/1niiHBo9LX6-I3KsEWYOi_s6cul80IYvK/view?usp=sharing)
DIS-VD    | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYJm3BqheaxNhdVoMt6X41gBgVnE4dulBwkp6pbOQtcIrQ?e=T6dtXm) | [Link](https://drive.google.com/file/d/1jhlZb3QyNPkc0o8nL3RWF0MLuVsVtJju/view?usp=sharing)
DIS-TE1   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcGYE_Gc0cVHoHi_qUtmsawB_5v9RSpJS5PIAPlLu6xo9A?e=Nu5mkQ) | [Link](https://drive.google.com/file/d/1iz8Y4uaX3ZBy42N2MIOkmNb0D5jroFPJ/view?usp=sharing)
DIS-TE2   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdhgMdbZ049GvMv7tNrjbbQB1wL9Ok85YshiXIkgLyTfkQ?e=mPA6Po) | [Link](https://drive.google.com/file/d/1DWSoWogTWDuS2PFbD1Qx9P8_SnSv2zTe/view?usp=sharing)
DIS-TE3   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EcxXYC_3rXxKsQBrp6BdNb4BOKxBK3_vsR9RL76n7YVG-g?e=2M0cse) | [Link](https://drive.google.com/file/d/1bIVSjsxCjMrcmV1fsGplkKl9ORiiiJTZ/view?usp=sharing)
DIS-TE4   | `data/Test_Dataset/...` | [Link](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdkG2SUi8flJvoYbHHOmvMABsGhkCJCsLLZlaV2E_SZimA?e=zlM2kC) | [Link](https://drive.google.com/file/d/1VuPNqkGTP1H4BFEHe807dTIkv8Kfzk5_/view?usp=sharing)

## Train & Evaluate

  * Train InSPyReNet
  ```
  # Single GPU
  python run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  
  # Multi GPUs with DDP (e.g., 4 GPUs)
  torchrun --standalone --nproc_per_node=4 run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose

  # Multi GPUs with DDP with designated devices (e.g., 2 GPUs - 0 and 1)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

  * Train with extra training datasets can be done by just changing [Train.Dataset.sets](https://github.com/plemeri/InSPyReNet/blob/main/configs/InSPyReNet_SwinB.yaml#L12) in the `yaml` config file, which is just simply adding more directories (e.g., HRSOD-TR, HRSOD-TR-LR, UHRSD-TR, ...):
   ```
   Train:
     Dataset:
         type: "RGB_Dataset"
         root: "data/RGB_Dataset/Train_Dataset"
         sets: ['DUTS-TR'] --> ['DUTS-TR', 'HRSOD-TR-LR', 'UHRSD-TR-LR']
   ```
  * Inference for test benchmarks
  ```
  python run/Test.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Evaluate metrics
  ```
  python run/Eval.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

  * All-in-One command (Train, Test, Eval in single command)
  ```
  # Single GPU
  python Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose

  # Multi GPUs with DDP (e.g., 4 GPUs)
  torchrun --standalone --nproc_per_node=4 Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose

  # Multi GPUs with DDP with designated devices (e.g., 2 GPUs - 0 and 1)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 Expr.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```


   * Please note that we only uploaded the low-resolution (LR) version of HRSOD and UHRSD due to their large image resolution. In order to use them, please download them from the original repositories (see references below), and change the directory names as we did to the LR versions.

## Inference on your own data
  + You can inference your own single image or images (.jpg, .jpeg, and .png are supported), single video or videos (.mp4, .mov, and .avi are supported), and webcam input (ubuntu and macos are tested so far).
  ```
  python run/Inference.py --config configs/InSPyReNet_SwinB.yaml --source [SOURCE] --dest [DEST] --type [TYPE] --gpu --jit --verbose
  ```

  + SOURCE: Specify your data in this argument.
    + Single image - `image.png`
    + Folder containing images - `path/to/img/folder`
    + Single video - `video.mp4`
    + Folder containing videos - `path/to/vid/folder`
    + Webcam input: `0` (may vary depends on your device.)
  + DEST (optional): Specify your destination folder. If not specified, it will be saved in `results` folder.
  + TYPE: Choose between `map, green, rgba, blur`
    + `map` will output saliency map only. 
    + `green` will change the background with green screen. 
    + `rgba` will generate RGBA output regarding saliency score as an alpha map. Note that this will not work for video and webcam input. 
    + `blur` will blur the background.
  + --gpu: Use this argument if you want to use GPU. 
  + --jit: Slightly improves inference speed when used. 
  + --verbose: Use when you want to visualize progress.