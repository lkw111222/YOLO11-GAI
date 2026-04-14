import torch
from thop import profile
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型（绝对路径）
    model = YOLO(r"YOLO-GAI\yolo11gai.pt").model
    model.eval()

    # 输入尺寸严格匹配训练尺寸：244×1350，RGB 3通道
    input_tensor = torch.randn(1, 3, 244, 1350)

    # 统计参数量和FLOPs
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    # 论文格式输出
    print("="*60)
    print(f"模型：YOLOv11x-seg")
    print(f"输入尺寸：3×244×1350")
    print(f"总参数量：{params / 1e6:.2f} M")
    print(f"总FLOPs：{flops / 1e9:.2f} G")
    print("="*60)