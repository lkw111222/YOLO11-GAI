# YOLO11-GAI
Intelligent Visual Recognition and Quantitative Characterization of Coal Mine Surrounding Rock Fractures Using Lightweight Deep Learning


# Environment Preparation
(1) Anaconda Environment Setup
Anaconda installation address: https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D
(2) Installing Core Dependencies
Install a specific version of Python and create an isolated virtual environment. Use the command: conda create -n yolo11_GAI python=3.9
Install the corresponding version of CUDA using the command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
(3) Others
For other installation methods, please refer to the Quick Start Guide and the complete Ultralytics documentation.

# Prepraration
git clone https://github.com/lkw111222/YOLO11-GAI
cd YOLO11-GAI
pip install -r requirements.txt

# Project Structure
YOLO11-GAI/
├── train.py             # 模型训练主脚本（核心）
├── val.py               # 模型验证脚本
├── FPS.py               # 模型推理速度测试脚本
├── yolo11gai.yaml       # 自定义模型配置文件
├── yolov11n.pt          # 预训练权重文件
├── sine.yaml            # 数据集配置文件
├── README.md            # 项目说明文档
└── runs/                # 训练结果输出目录（自动生成）
    └── train/           # 训练日志、权重、指标文件
└── sine/                # 数据集
    └── images/          # 数据集图像存放目录
        └── train/       # 训练集图像存放目录
        └── val/         # 验证集图像存放目录
    └── labels/          # label存放目录
        └── train/       # 训练集图像对应label存放目录
        └── val/         # 验证集图像对应label存放目录

# Acknowledgements
We sincerely thank the outstanding Ultralytics team for their tremendous support of the models we developed in YOLO11-GAI.
We also extend our gratitude to all other contributors of the Ultralytics platform team, who have contributed such impressive models to the community.

# Citation
If you find this project useful, please consider citing:
@article{YOLO11-GAI,
  title={YOLO11-GAI},
  author={Kangwei Liu, Chuanzhi Ning, Mingyang Wang, Yaocong Hu, Guoyang Wan, Bingyou Liu},
  journal={Visual Computer},
  year={2026}
}

@inproceedings{YOLO11-GAI,
  title={Intelligent Visual Recognition and Quantitative Characterization of Coal Mine Surrounding Rock Fractures Based on Lightweight Deep Learning}, 
  author={Kangwei Liu, Chuanzhi Ning, Mingyang Wang, Yaocong Hu, Guoyang Wan, Bingyou Liu},
  booktitle={Visual Computer},
  year={2026}
}
