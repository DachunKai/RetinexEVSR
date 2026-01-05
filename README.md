# [RetinexEVSR (AAAI 2026)](https://arxiv.org/pdf/2504.13042)

Official PyTorch implementation for the "Seeing the Unseen: Zooming in the Dark with Event Cameras" paper (AAAI 2026).
<!-- 
<p align="center">
    <a href="https://arxiv.org/pdf/2504.13042" target="_blank">📃 Paper</a>
</p> -->

**Authors**: [Dachun Kai](https://github.com/DachunKai/)<sup>[:email:️](mailto:dachunkai@mail.ustc.edu.cn)</sup>, [Zeyu Xiao](https://dblp.org/pid/276/3139.html), Huyue Zhu, Jiaxiao Wang, [Yueyi Zhang](https://scholar.google.com.hk/citations?user=LatWlFAAAAAJ&hl=zh-CN&oi=ao), [Xiaoyan Sun](https://scholar.google.com/citations?user=VRG3dw4AAAAJ&hl=zh-CN), *University of Science and Technology of China*

**Feel free to ask questions. If our work helps, please don't hesitate to give us a :star:!**

## :rocket: News
- [x] 2026/01/05: Make repository public
- [x] 2025/11/17: Release pretrained models and test sets for quick testing
- [x] 2025/11/17: Video demos released
- [x] 2025/11/15: Initialize the repository
- [x] 2025/11/08: :tada: :tada: Our paper was accepted in AAAI'2026

## :bookmark: Table of Contents
1. [Video Demos](#video-demos)
2. [Code](#code)
3. [Citation](#citation)
4. [Contact](#contact)
5. [License and Acknowledgement](#license-and-acknowledgement)

## :fire: Video Demos
Visual results of $4\times$ upsampling on the real-world [SDE](https://github.com/EthanLiang99/EvLight) and [RELED](https://github.com/intelpro/ELEDNet) datasets, transforming low-light LR videos into well-lit HR videos.

https://github.com/user-attachments/assets/92108fe3-b72d-4551-b908-21ede093508b

https://github.com/user-attachments/assets/48872d3a-36e2-4b20-a273-fd6e030e8afd

https://github.com/user-attachments/assets/f11eae03-3641-42ce-b959-097e5c4a7c72

## Code
### Installation
* Dependencies: [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), [CUDA Toolkit 11.1.1](https://developer.nvidia.com/cuda-11.1.1-download-archive), [torch 1.10.2+cu111](https://download.pytorch.org/whl/cu111/torch-1.10.2%2Bcu111-cp37-cp37m-linux_x86_64.whl), and [torchvision 0.11.3+cu111](https://download.pytorch.org/whl/cu111/torchvision-0.11.3%2Bcu111-cp37-cp37m-linux_x86_64.whl).

* Run in Conda (**Recommended**)

    ```bash
    conda create -y -n RetinexEVSR python=3.7
    conda activate RetinexEVSR
    pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
    git clone https://github.com/DachunKai/RetinexEVSR
    cd RetinexEVSR && pip install -r requirements.txt && python setup.py develop
    ```
* Run in Docker :clap:

  Note: **We use the same docker image as our previous work [EvTexture](https://github.com/DachunKai/EvTexture)**.

  [Option 1] Directly pull the published Docker image we have provided from [Alibaba Cloud](https://cr.console.aliyun.com/cn-hangzhou/instances).
  ```bash
  docker pull registry.cn-hangzhou.aliyuncs.com/dachunkai/evtexture:latest
  ```

  [Option 2] We also provide a [Dockerfile](https://github.com/DachunKai/RetinexEVSR/blob/main/docker/Dockerfile) that you can use to build the image yourself.
  ```bash
  cd RetinexEVSR && docker build -t retinexevsr ./docker
  ```
  The pulled or self-built Docker image contains a complete conda environment. After running the image, you can mount your data and operate within this environment.
  ```bash
  source activate RetinexEVSR && cd RetinexEVSR && python setup.py develop
  ```
### Test
1. Download the pretrained models from ([Releases](https://github.com/DachunKai/RetinexEVSR/releases) / [Google Drive](https://drive.google.com/drive/folders/1kHF_w1RHZLUOvo7GKptnLOXxAi7MDuqA?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/1qGWe_O_fXTVfdFJKLXkZcw?pwd=n8hg) (n8hg)) and place them to `experiments/pretrained_models/RetinexEVSR/`. The network architecture code is in [retinexevsr_arch.py](https://github.com/DachunKai/RetinexEVSR/blob/main/basicsr/archs/retinexevsr_arch.py).
    - Synthetic dataset models:
      * *RetinexEVSR_SDSD_Indoor_BIx4.pth*: trained on [SDSD Indoor](https://github.com/JIA-Lab-research/SDSD) dataset.
      * *RetinexEVSR_SDSD_Outdoor_BIx4.pth*: trained on [SDSD Outdoor](https://github.com/JIA-Lab-research/SDSD) dataset.
    - Real-world dataset model:
      * *RetinexEVSR_SDE_Indoor_BIx4.pth*: trained on [SDE Indoor](https://github.com/EthanLiang99/EvLight) dataset.
      * *RetinexEVSR_SDE_Outdoor_BIx4.pth*: trained on [SDE Outdoor](https://github.com/EthanLiang99/EvLight) dataset.
      * *RetinexEVSR_RELED_BIx4.pth*: trained on [RELED](https://github.com/intelpro/ELEDNet) dataset.

2. Download the preprocessed test sets (including events) for [SDSD](https://github.com/JIA-Lab-research/SDSD), [SDE](https://github.com/EthanLiang99/EvLight), and [RELED](https://github.com/intelpro/ELEDNet) from ([Google Drive](https://drive.google.com/drive/folders/1kHF_w1RHZLUOvo7GKptnLOXxAi7MDuqA?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/1qGWe_O_fXTVfdFJKLXkZcw?pwd=n8hg) (n8hg)), and place them to `datasets/`.
    * *SDSD*: HDF5 files containing preprocessed test datasets for SDSD_Indoor and SDSD_Outdoor.
    * *SDE*: HDF5 files containing preprocessed test datasets for SDE_Indoor and SDE_Outdoor.
    * *RELED*: HDF5 files containing preprocessed test datasets for RELED.

3. Run the following command:
    We use 8*4090 to test, which is explicitly quicker than two gpus.
    * Test on SDSD Indoor for 4x Low-Light VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/RetinexEVSR/test_RetinexEVSR_SDSD_IN_x4.yml
      ```
    * Test on SDSD Outdoor for 4x Low-Light VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/RetinexEVSR/test_RetinexEVSR_SDSD_OUT_x4.yml
      ```
    * Test on SDE Indoor for 4x Low-Light VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/RetinexEVSR/test_RetinexEVSR_SDE_IN_x4.yml
      ```
    * Test on SDE Outdoor for 4x Low-Light VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/RetinexEVSR/test_RetinexEVSR_SDE_OUT_x4.yml
      ```
    * Test on RELED for 4x Low-Light VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/RetinexEVSR/test_RetinexEVSR_RELED_x4.yml
      ```
    This will generate the inference results in `results/`. The output results on SDSD, SDE and RELED datasets can be downloaded from ([Releases](https://github.com/DachunKai/RetinexEVSR/releases) / [Google Drive](https://drive.google.com/drive/folders/1kHF_w1RHZLUOvo7GKptnLOXxAi7MDuqA?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/1qGWe_O_fXTVfdFJKLXkZcw?pwd=n8hg) (n8hg)).

4. Test the number of parameters, runtime, and FLOPs:
    ```bash
    python test_scripts/test_params_runtime.py
    ```

### Input Data Structure
* Both video and event data are required as input. We package each video and its event data into an [HDF5](https://docs.h5py.org/en/stable/quick.html#quick) file.
* The Low-Light (LR) HDF5 file contains `images` and `voxels`.
* The Ground-Truth (GT) HDF5 file contains `images`.

* Example: The structure of an HDF5 file:
  ```
  ClipName.h5
  ├── images
  │   ├── 000000 # frame, ndarray, [H, W, C]
  │   ├── ...
  ├── voxels
  │   ├── 000000 # event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ```

## Citation
If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{kai2026seeing,
  title={Seeing the Unseen: Zooming in the Dark with Event Cameras},
  author={Kai, Dachun and Xiao, Zeyu and Zhu, Huyue and Wang, Jiaxiao and Zhang, Yueyi and Sun, Xiaoyan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Contact
If you have any questions, please contact [Dachun Kai](mailto:dachunkai@mail.ustc.edu.cn).

## License and Acknowledgement
This project is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Our work builds significantly upon our previous project [EvTexture](https://github.com/DachunKai/EvTexture) and [Ev-DeblurVSR](https://github.com/DachunKai/Ev-DeblurVSR). We would also like to sincerely thank the developers of [BasicSR](https://github.com/XPixelGroup/BasicSR), an open-source toolbox for image and video restoration tasks. Additionally, we appreciate the inspiration and code provided by [event_utils](https://github.com/TimoStoff/event_utils).
