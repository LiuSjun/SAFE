import numpy as np
import cv2
import rasterio
from segment_anything import sam_model_registry, SamPredictor
import torch
from UNIT import img2numpy, numpy2img
import os

# 输入和输出图像的名称设置
image_path = r"D:\Resouce\BurnedArea\Data\Gobal\Source\XC\Sentinel2_BurnedAreas__22773_6213Xichang.tif"  # 输入 TIFF 图像路径
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_dir = os.path.dirname(image_path)

output_stretched_rgb_image = os.path.join(output_dir, f"{image_name}_stretched_rgb_image.png")
output_best_segmentation_result = os.path.join(output_dir, f"{image_name}_best_segmentation_result_l.png")
output_nbr_image = os.path.join(output_dir, f"{image_name}_nbr_image.png")
output_burned_area_nbr = os.path.join(output_dir, f"{image_name}_burned_area_nbr.png")
output_bais2_image = os.path.join(output_dir, f"{image_name}_bais2_image.png")
output_burned_area_bais2 = os.path.join(output_dir, f"{image_name}_burned_area_bais2.png")

# 加载模型
model_type = "vit_l"  # "vit_h"或 "vit_l", "vit_b"
checkpoint_path = "sam_vit_l_0b3195.pth"  # 或 "sam_vit_h_4b8939", "sam_vit_b_01ec64", "sam_vit_l_0b3195"
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
def envi_linear_stretch(image, p_low=2, p_high=98):
    # 计算图像的低百分位和高百分位
    p_low_val = np.percentile(image, p_low)
    p_high_val = np.percentile(image, p_high)

    # 拉伸图像
    stretched_image = (image - p_low_val) / (p_high_val - p_low_val)
    stretched_image = np.clip(stretched_image, 0, 1)  # 保留在0到1的范围内
    return stretched_image

# 加载影像
image = img2numpy(image_path)

# 提取前三个波段
image_rgb = np.nan_to_num(image[:3].transpose((1, 2, 0)))

# 对RGB图像进行ENVI风格的线性拉伸
image_rgb_stretched = envi_linear_stretch(image_rgb)

# 将拉伸后的图像缩放到0-255范围以便保存
image_rgb_stretched_display = (image_rgb_stretched * 255).astype(np.uint8)

# 保存拉伸后的RGB图像
cv2.imwrite(output_stretched_rgb_image, image_rgb_stretched_display)

# 提取第七个波段
band_7 = image[-1]

# 生成提示框（外控矩形）
def generate_bbox_from_band(band):
    non_zero_points = np.argwhere(band > 0)
    if len(non_zero_points) > 0:
        y_min, x_min = np.min(non_zero_points, axis=0)
        y_max, x_max = np.max(non_zero_points, axis=0)
        # 生成提示框（外控矩形），并将其转换为 NumPy 数组
        return np.array([[x_min, y_min, x_max, y_max]])
    else:
        raise ValueError("第七个波段中没有值大0的点")

def generate_bbox_from_band_Pixl(band, expand_pixels=200):
    non_zero_points = np.argwhere(band > 0)
    if len(non_zero_points) > 0:
        y_min, x_min = np.min(non_zero_points, axis=0)
        y_max, x_max = np.max(non_zero_points, axis=0)

        # 扩展外控矩形
        y_min = max(y_min - expand_pixels, 0)
        x_min = max(x_min - expand_pixels, 0)
        y_max = min(y_max + expand_pixels, band.shape[0] - 1)
        x_max = min(x_max + expand_pixels, band.shape[1] - 1)

        return np.array([[x_min, y_min, x_max, y_max]])
    else:
        raise ValueError("第七个波段中没有值大0的点")


box = generate_bbox_from_band_Pixl(band_7)

# 创建预测器并进行分割
predictor = SamPredictor(sam)
predictor.set_image(image_rgb_stretched_display)

# 使用外控矩形作为提示框进行分割
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
    cv2.imwrite(output_best_segmentation_result, best_mask_image_bgr)
else:
    print("没有生成任何mask")

# 计算 NBR
def calculate_nbr(nir, swir):
    return (nir - swir) / (nir + swir)

band_4 = image[3]  # 近红外
band_11 = image[4]  # 短波红外1

# 计算 NBR
nbr = calculate_nbr(band_4, band_11)
nbr = np.nan_to_num(nbr)

# 处理 NBR（例如线性拉伸）
p_low_val = np.percentile(nbr, 2)
p_high_val = np.percentile(nbr, 98)
nbr_stretched = (nbr - p_low_val) / (p_high_val - p_low_val)
nbr_stretched = np.clip(nbr_stretched, 0, 1)
nbr_display = (nbr_stretched * 255).astype(np.uint8)

# 保存 NBR 图像
cv2.imwrite(output_nbr_image, nbr_display)

# 设置阈值提取烧毁区域
threshold = 200
burned_area = (nbr_display >= threshold).astype(np.uint8) * 255
cv2.imwrite(output_burned_area_nbr, burned_area)

# 计算 BAIS2
def calculate_bais2(band_4, band_6, band_7, band_8, band_12):
    return ( 1 - np.sqrt(( band_6 * band_7 * band_8)) / band_4) / ( (band_12 - band_8) / np.sqrt(( band_12 + band_8))+ 1)

band_4 = image[2]  # 短波红夔
band_6 = image[3]  # 短波红夔
band_71 = image[4]  # 短波红夔
band_8 = image[5]  # 短波红夔
band_12 = image[7]  # 短波红夔

# 计算 BAIS2
bais2 = calculate_bais2(band_4, band_6, band_71, band_8, band_12)
bais2 = np.nan_to_num(bais2)

# 处理 BAIS2（例如线性拉伸）
p_low_val = np.percentile(bais2, 2)
p_high_val = np.percentile(bais2, 98)
bais2_stretched = (bais2 - p_low_val) / (p_high_val - p_low_val)
bais2_stretched = np.clip(bais2_stretched, 0, 1)
bais2_display = (bais2_stretched * 255).astype(np.uint8)

# 保存 BAIS2 图像
cv2.imwrite(output_bais2_image, bais2_display)

# 设置阈值提取烧毁区域
threshold = 120
burned_area_bais2 = (bais2_display <= threshold).astype(np.uint8) * 255
cv2.imwrite(output_burned_area_bais2, burned_area_bais2)
