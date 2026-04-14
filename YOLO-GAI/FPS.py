from ultralytics import YOLO
import time

# 加载训练好的模型（替换为你的模型路径）
model = YOLO(r"C:\daima\YOLO-GAI\yolo11gai.pt")

# 替换为你实际的验证集图像路径（从左侧文件夹找，比如“4原数据集”下的val/images）
test_image_path = r"YOLO-GAI\sine\images\val"  # 需根据实际路径修改

# 统计100次推理的平均时间
num_runs = 100
total_time = 0

# 循环推理
for _ in range(num_runs):
    start = time.time()
    # 推理（verbose=False关闭输出）
    model.predict(source=test_image_path, imgsz=640, verbose=False)
    total_time += (time.time() - start)

# 计算FPS
fps = num_runs / total_time
print(f"模型在验证集上的平均FPS：{fps:.2f}")