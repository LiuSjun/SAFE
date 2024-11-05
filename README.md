# SAFE
Segmentation of Any Fire Event (SAFE): A rapid and high-precision approach for burned area extraction using sentinel-2 imagery

> **[NUIST, Leo-RS]**
> 
> Contributors: [Shuaijun Liu](https://alex1234.github.io/)
> 
> Resources: [[`Academic Paper`]] [[`Demo`]]

<p align="center">
  <img src="flowchart.png?raw=true" width="50.25%" />
</p>

## Overview
**SAFE (Segmentation of Any Fire Event)** is an adaptive model designed for segmentation of burned area from remote sensing imagery. Building upon the 'Segment Anything Model', it boasts enhanced zero-shot performance in remote sensing image analysis.

<p align="center">
  <img src="SAFE.png?raw=true" width="37.25%" />
</p>

## Installation and Requirements

### System Requirements
- Python 3.8+
- PyTorch 1.7.0+
- CUDA 11.0+ (Recommended)

### Installation Instructions
SAFE can be easily installed via pip or by cloning the repository.

### Additional Dependencies
For mask post-processing and running example notebooks, additional packages are required.

### Prerequisites
- numpy 1.24.3
- torchvision 0.8+
- GDAL, OpenCV
- [Albumentations](https://pypi.org/project/albumentations/) 1.3.1+

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install SAFE:

```
pip install SAFE.git
```

or clone the repository locally and install with

```
git clone git@github.com:SAFE.git
cd SAFE; pip install -e .
```

The following optional dependencies are necessary for mask post-processing,`jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Getting Started with SAFE

First download STAMP. Then the model can be used in just a few lines to get masks:

```
from SAFE import auotSAFE
safe = auotSAFE["<model_type>"]
predictor = safe(pic)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

For detailed examples, see our [notebooks](/notebooks/SAFE_example.ipynb).

<p float="left">
  <img src="pic/Fig3.png?raw=true" width="36.1%" />
  <img src="pic/Fig4.png?raw=true" width="48.9%" />
</p>

## Demonstrations

### Software Demo
Explore the `SAFE` one-page app for intuitive mask prediction. Detailed instructions are available in [`SAFEWindow.md`](https://github.com/LiuSjun/SAFE/README.md).

#### Demo Steps
1. **Start the Demo**: Double-click 'SAFE.exe'.
   <p align="center">
     <img src="pic/Step1.gif?raw=true" width="50.25%" />
   </p>

2. **Select and Open Image**.
   <p align="center">
     <img src="pic/step2.gif?raw=true" width="50.25%" />
   </p>

3. **Import or Auto-Select Processing Area**.
   <p align="center">
     <img src="pic/step3.gif?raw=true" width="50.25%" />
   </p>

4. **Extract Missing BAs (Burn Area)** (manually or automatically).
   <p align="center">
     <img src="pic/step4.gif?raw=true" width="50.25%" />
     <img src="pic/step5.gif?raw=true" width="50.25%" />
   </p>

## License and Citation

### License
SAFE is licensed under [beta 3.0.2](LICENSE).

### How to Cite
If you use SAFE in your research, please use the following BibTeX entry.

```
@article{kirillov20234safe,
  title={Segmentation of Any Fire Event (SAFE): A Rapid and High-Precision Approach for Burned Area Extraction Using Sentinel-2 Imagery},
  author={Liu Shuaijun, Chen Hui, Shu Hongtao, Huang Ping, Chen Yang, Zhan Tianyu},
  journal={arXiv:2410.02963},
  year={2024}
}
}
