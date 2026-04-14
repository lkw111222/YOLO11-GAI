from ultralytics import YOLO
import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize

# 加载训练好的模型
model = YOLO('yolov11_training-seg/exp/weights/best.pt')  # 替换为你的模型路径

# 推理单张图片
image_path = 'sine/images/train/3.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)
results = model(image_path)

# 创建与原图同样大小的白色背景图像
mask_image = np.ones_like(image) * 255  # 白色背景

# 处理分割结果
for r in results:
    # 获取分割掩码
    if hasattr(r, 'masks') and r.masks is not None:
        # 遍历所有检测到的对象
        for mask in r.masks.data:
            # 将掩码转换为numpy数组
            mask_np = mask.cpu().numpy()

            # 调整掩码大小以匹配原图
            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            # 将分割区域设置为黑色 (0, 0, 0)
            # 注意：mask_resized是二值图像，需要转换为布尔类型
            mask_bool = mask_resized > 0.5  # 转换为布尔掩码
            mask_image[mask_bool] = [0, 0, 0]  # 设置为黑色

# 保存结果
cv2.imwrite('result-seg-mask.jpg', mask_image)


# 提取裂隙中心线
def extract_centerline(binary_mask):
    """
    从二值化掩码中提取中心线
    """
    # 转换为灰度图并二值化
    if len(binary_mask.shape) == 3:
        gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_mask

    # 反转图像：黑色为裂隙区域，白色为背景
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 转换为布尔数组
    binary_bool = binary > 127

    # 细化处理获取中心线
    skeleton = skeletonize(binary_bool)

    # 找到中心线上的所有点
    centerline_points = np.column_stack(np.where(skeleton))

    # 调整坐标顺序为 (x, y)
    centerline_points = np.flip(centerline_points, axis=1)

    return centerline_points, skeleton


# 获取中心线
centerline_points, skeleton = extract_centerline(mask_image)

# 保存中心线图像
skeleton_image = np.zeros_like(mask_image)
if len(skeleton_image.shape) == 3:
    skeleton_image[skeleton] = [255, 255, 255]  # 白色中心线
else:
    skeleton_image[skeleton] = 255

cv2.imwrite('result-centerline.jpg', skeleton_image)


# 对中心点进行排序（如果需要按顺序排列）
def sort_centerline_points(points):
    """
    对中心线点进行排序，使其按顺序排列
    """
    if len(points) < 2:
        return points

    # 使用简单的最近邻排序
    sorted_points = [points[0]]
    remaining_points = list(points[1:])

    while remaining_points:
        last_point = sorted_points[-1]
        # 找到距离最后一个点最近的点
        distances = [np.linalg.norm(np.array(last_point) - np.array(p)) for p in remaining_points]
        nearest_idx = np.argmin(distances)
        sorted_points.append(remaining_points.pop(nearest_idx))

    return np.array(sorted_points)


# 排序中心点
sorted_centerline_points = sort_centerline_points(centerline_points)

# 打印中心点信息
print(f"检测到 {len(centerline_points)} 个中心点")
print("前10个中心点坐标:")
for i, point in enumerate(sorted_centerline_points[:10]):
    print(f"  点 {i + 1}: ({point[0]}, {point[1]})")

# 在原图上绘制中心点
result_with_points = mask_image.copy()
for point in sorted_centerline_points[::max(1, len(sorted_centerline_points) // 20)]:  # 每隔一定距离绘制一个点
    cv2.circle(result_with_points, tuple(point.astype(int)), 3, (0, 0, 255), -1)

cv2.imwrite('result-centerline-points.jpg', result_with_points)

print("推理完成，分割掩码已保存为 result-seg-mask.jpg")
print("中心线已保存为 result-centerline.jpg")
print("带中心点的结果已保存为 result-centerline-points.jpg")