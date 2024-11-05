import numpy as np
import cv2
import rasterio
from segment_anything_2 import sam_model_registry, SamPredictor
import torch
from UNIT import img2numpy, numpy2img

# 加载模型
model_type = "vit_b"  # 或 "vit_l", "vit_b"
checkpoint_path = "sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint_path)
sam.to(device)

# 读取多波段遥感影像
def load_multiband_image(image_path):
    with rasterio.open(image_path) as src:
        bands = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            bands.append(band)
    return bands

# ENVI风格的线性拉伸（2%）
def envi_linear_stretch(image, p_low=5, p_high=95):
    # 计算图像的低百分位和高百分位
    p_low_val = np.percentile(image, p_low)
    p_high_val = np.percentile(image, p_high)

    # 拉伸图像
    stretched_image = (image - p_low_val) / (p_high_val - p_low_val)
    stretched_image = np.clip(stretched_image, 0, 1)  # 保留在0到1的范围内
    return stretched_image

# 加载影像
image_path = r"D:\Resouce\BurnedArea\Data\Gobal\drive-download-20241022T073836Z-001\4\Sentinel2_BurnedAreas_Final_1.tif"  # TIFF 图像路径
image = img2numpy(image_path)

# 提取前三个波段
image_rgb = np.nan_to_num(image[:3].transpose((1, 2, 0)))

# 对RGB图像进行ENVI风格的线性拉伸
image_rgb_stretched = envi_linear_stretch(image_rgb)

# 将拉伸后的图像缩放到0-255范围以便保存
image_rgb_stretched_display = (image_rgb_stretched * 255).astype(np.uint8)

# 提取第七个波段
band_7 = image[6]

# 生成提示框（外接矩形）
def generate_bbox_from_band(band):
    non_zero_points = np.argwhere(band > 0)
    if len(non_zero_points) > 0:
        y_min, x_min = np.min(non_zero_points, axis=0)
        y_max, x_max = np.max(non_zero_points, axis=0)
        # 生成提示框（外接矩形），并将其转换为 NumPy 数组
        return np.array([[x_min, y_min, x_max, y_max]])
    else:
        raise ValueError("第七个波段中没有值大于0的点")
def generate_bbox_from_band_Pixl(band, expand_pixels=200):
    non_zero_points = np.argwhere(band > 0)
    if len(non_zero_points) > 0:
        y_min, x_min = np.min(non_zero_points, axis=0)
        y_max, x_max = np.max(non_zero_points, axis=0)

        # 扩展外接矩形
        y_min = max(y_min - expand_pixels, 0)
        x_min = max(x_min - expand_pixels, 0)
        y_max = min(y_max + expand_pixels, band.shape[0] - 1)
        x_max = min(x_max + expand_pixels, band.shape[1] - 1)

        return np.array([[x_min, y_min, x_max, y_max]])
    else:
        raise ValueError("第七个波段中没有值大于0的点")

box = generate_bbox_from_band_Pixl(band_7)

# 创建预测器并进行分割
predictor = SamPredictor(sam)
predictor.set_image(image_rgb_stretched_display)

# 使用外接矩形作为提示框进行分割
masks, scores, logits = predictor.predict(
    box=box,
    multimask_output=True,
)

# 找到得分最高的mask
if len(scores) > 0:
    best_mask_index = np.argmax(scores)
    best_mask = masks[best_mask_index]
    best_mask_image = (best_mask > 0.5).astype(np.uint8) * 255
    best_mask_image_bgr = cv2.cvtColor(best_mask_image, cv2.COLOR_GRAY2BGR)  # 转换为 BGR 格式以便保存
    cv2.imwrite("best_segmentation_result.png", best_mask_image_bgr)
else:
    print("没有生成任何mask")

# 计算 NBR
def calculate_nbr(nir, swir):
    return (nir - swir) / (nir + swir)

band_4 = image[3]  # 近红外
band_11 = image[4]  # 短波红外 1

# 计算 NBR
nbr = calculate_nbr(band_4, band_11)
nbr = np.nan_to_num(nbr)

# 处理 NBR（例如线性拉伸）
p_low_val = np.percentile(nbr, 2)
p_high_val = np.percentile(nbr, 98)
nbr_stretched = (nbr - p_low_val) / (p_high_val - p_low_val)
nbr_stretched = np.clip(nbr_stretched, 0, 1)
nbr_display = (nbr_stretched * 255).astype(np.uint8)

# 设置阈值提取烧毁区域
threshold = 50
burned_area = (nbr_display < threshold).astype(np.uint8) * 255

# 创建四宫格图像
h, w, _ = image_rgb_stretched_display.shape
grid_image = np.zeros((h*2, w*2, 3), dtype=np.uint8)

# 1. 原始RGB图像
grid_image[:h, :w] = cv2.cvtColor(image_rgb_stretched_display, cv2.COLOR_RGB2BGR)

# 2. 在RGB上覆盖第七个波段的部分
overlay = image_rgb_stretched_display.copy()
overlay[band_7 > 0] = [0, 0, 255]  # 用红色标识
grid_image[:h, w:] = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

# 3. 标识最小外接矩形
rect_image = image_rgb_stretched_display.copy()
[x_min, y_min, x_max, y_max ] = box[0]
cv2.rectangle(rect_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), thickness=cv2.FILLED)  # 白色填充矩形
grid_image[h:, :w] = cv2.cvtColor(burned_area, cv2.COLOR_RGB2BGR)

# 4. 标识得分最高的Mask
if best_mask_image is not None:
    mask_overlay = image_rgb_stretched_display.copy()
    mask_overlay[best_mask_image > 0] = [0, 0, 255]  # 用红色标识
    grid_image[h:, w:] = cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR)
else:
    grid_image[h:, w:] = cv2.cvtColor(image_rgb_stretched_display, cv2.COLOR_RGB2BGR)  # 备用显示

# 保存四宫格图像
cv2.imwrite("four_grid_image.png", grid_image)

# 保存拉伸后的RGB图像
cv2.imwrite("stretched_rgb_image.png", cv2.cvtColor(image_rgb_stretched_display, cv2.COLOR_RGB2BGR))
