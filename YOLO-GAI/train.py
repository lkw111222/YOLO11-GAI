from ultralytics import YOLO

model = YOLO('yolo11gai.yaml').load('yolo11gai.pt')  # 从YAML文件构建模型并加载预训练权重
results = model.train(
    data='YOLO-GAI/sine.yaml',  # 数据集配置文件路径，需根据你的数据集修改
    epochs=300,  # 训练轮次
    imgsz=640,  # 输/入图像尺寸
    batch=16,  # 批次大小，根据GPU内存调整
    device='0',  # 训练设备，0表示第一块GPU，'cpu'表示使用CPU
    workers=0,  # 数据加载工作线程数
    save=True,  # 是否保存模型
    save_period=10,  # 每10轮保存一次模型
    project='yolov11_training',  # 项目名称
    name='exp',  # 实验名称
    optimizer='SGD',    # 优化器，可选 'SGD', 'Adam', 'AdamW'
)

# 训练完成后在验证集上评估模型
# metrics = model.val()