import json
import os
from PIL import Image

# -------------------------- 配置路径（改成你实际的路径） --------------------------
json_dir = r"yolo-gai-main\json"  # 你的JSON文件所在目录
output_label_dir = r"C:\daima\yolo-gai-main\txt"  # 输出YOLO格式.txt的目录
image_dir = r"C:\daima\yolo-gai-main\images"  # 你的图像文件所在目录（比如1-1.jpg在这里）

# -------------------------- 无需修改以下内容 --------------------------
os.makedirs(output_label_dir, exist_ok=True)

# 类别映射（你的数据集类别名→YOLO类别ID）
class_map = {"sine": 0}  # 把LabelMe里的标签名改成你实际用的（比如你的标签是“裂缝”，这里就写"裂缝":0）

for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue
    json_path = os.path.join(json_dir, json_file)

    # 读取JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 【关键修改】忽略JSON里的imagePath，直接用JSON文件名匹配图像（比如1-1.json对应1-1.jpg）
    img_name = os.path.splitext(json_file)[0] + ".jpg"  # 假设图像是.jpg格式，如果是.png改成.png
    img_path = os.path.join(image_dir, img_name)

    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"警告：图像{img_path}不存在，跳过该JSON文件")
        continue

    # 获取图像尺寸
    img = Image.open(img_path)
    img_w, img_h = img.size

    # 转换标注
    yolo_lines = []
    for shape in data["shapes"]:
        class_name = shape["label"]
        if class_name not in class_map:
            print(f"警告：类别{class_name}不在class_map中，跳过")
            continue
        class_id = class_map[class_name]

        # 多边形坐标归一化
        points = shape["points"]
        norm_points = []
        for x, y in points:
            norm_x = x / img_w
            norm_y = y / img_h
            norm_points.extend([norm_x, norm_y])

        # 生成YOLO格式行
        yolo_line = f"{class_id} " + " ".join(map(str, norm_points))
        yolo_lines.append(yolo_line)

    # 保存为txt
    txt_name = os.path.splitext(json_file)[0] + ".txt"
    txt_path = os.path.join(output_label_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

print("转换完成！")