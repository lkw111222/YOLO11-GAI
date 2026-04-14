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

├── train.py             # Main Script for Model Training (Core)

├── val.py               # Model Validation Script

├── FPS.py               # Model Inference Speed Test Script

├── yolo11gai.yaml       # Model Inference Speed Testing Script

├── yolov11n.pt          # Pre-trained weight file

├── sine.yaml            # Dataset Configuration File

├── README.md            # Project Specification Document

└── runs/                # Training Results Output Directory (Automatically Generated)  
    └── train/           # Training logs, weights, and metric files
    
└── sine/  

    └── images/          # Dataset Image Storage Directory
        └── train/       # Training set image storage directory  
        └── val/         # Validation set image storage directory
        
    └── labels/          # Label Storage Directory   
        └── train/       # Directory for storing labels corresponding to training set images     
        └── val/         # Directory for storing labels corresponding to validation set images

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
