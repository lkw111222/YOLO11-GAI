from ultralytics import YOLO
import cv2
import os
import glob

# ===================== 核心配置 =====================
MODEL_PATH = 'C:/daima/ultralytics-main-sine/runs/detect/train42/weights/best.pt'
INPUT_IMAGE_DIR = 'sine/images/train'
OUTPUT_RESULT_DIR = 'detection_results'
SUPPORTED_FORMATS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')

# 自定义类别颜色字典（key=类别ID，value=RGB颜色）
# 先查看模型的类别ID：print(model.names)，比如0=sine，就对应0:(255,0,0)
CLASS_COLORS = {
    0: (255, 255, 0),  # 类别0（sine）→ 红色
    # 1: (0, 255, 0),  # 若有类别1→绿色，按需添加
    # 2: (0, 0, 255),  # 若有类别2→蓝色，按需添加
}


# ===================== 核心功能实现 =====================
def batch_detect_and_save():
    print(f"正在加载模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    # 打印模型类别，方便核对ID
    print(f"模型检测类别：{model.names}")

    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)
    print(f"结果将保存到：{os.path.abspath(OUTPUT_RESULT_DIR)}")

    image_paths = []
    for fmt in SUPPORTED_FORMATS:
        image_paths.extend(glob.glob(os.path.join(INPUT_IMAGE_DIR, fmt)))

    if not image_paths:
        print(f"❌ 错误：在 {INPUT_IMAGE_DIR} 中未找到任何图片文件！")
        return

    print(f"✅ 共找到 {len(image_paths)} 张待检测图片")

    for idx, img_path in enumerate(image_paths, 1):
        try:
            img_name = os.path.basename(img_path)
            print(f"\n[{idx}/{len(image_paths)}] 正在检测：{img_name}")

            results = model(img_path)

            for r in results:
                # 手动绘制锚框（自定义颜色核心）
                im = r.orig_img.copy()  # 复制原始图片
                boxes = r.boxes  # 获取检测框信息

                if boxes is not None:
                    for box in boxes:
                        # 获取锚框坐标（xyxy格式）
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # 获取类别ID和置信度
                        cls_id = int(box.cls[0])
                        conf = box.conf[0].item()
                        cls_name = model.names[cls_id]

                        # 选择颜色：优先用自定义颜色，无则默认蓝色
                        color = CLASS_COLORS.get(cls_id, (0, 0, 255))
                        # 绘制锚框
                        cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)
                        # 绘制类别+置信度标签
                        label = f"{cls_name} {conf:.2f}"
                        # 标签背景（避免文字和图片重叠看不清）
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                        cv2.rectangle(im, (x1, label_y - label_size[1] - 5),
                                      (x1 + label_size[0], label_y + 5), (0, 0, 0), -1)
                        cv2.putText(im, label, (x1, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 保存自定义绘制后的图片
                save_path = os.path.join(OUTPUT_RESULT_DIR, img_name)
                cv2.imwrite(save_path, im)
                print(f"   ✅ 检测完成，结果保存至：{save_path}")

        except Exception as e:
            print(f"   ❌ 检测 {img_name} 时出错：{str(e)}")

    print(f"\n🎉 批量检测完成！所有结果已保存到：{os.path.abspath(OUTPUT_RESULT_DIR)}")


if __name__ == "__main__":
    batch_detect_and_save()