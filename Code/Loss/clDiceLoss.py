import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

def dilate_mask(mask, dilation=2):
    """
    mask: [B,1,H,W] 0/1
    dilation: 膨胀半径
    """
    kernel_size = 2 * dilation + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    # 只要邻域内有像素就设为1
    out = F.conv2d(mask, kernel, padding=dilation)
    return (out > 0).float()

# GPT写法，
# 膨胀腐蚀操作更简单，是一种简化计算更快的近似，
# 同时迭代次数多有利于骨架精细提取
# def soft_erode(img):
#     p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
#     p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
#     return torch.min(p1, p2)
#
#
# def soft_dilate(img):
#     p1 = F.max_pool2d(img, (3, 1), (1, 1), (1, 0))
#     p2 = F.max_pool2d(img, (1, 3), (1, 1), (0, 1))
#     return torch.max(p1, p2)
#
#
# def soft_open(img):
#     return soft_dilate(soft_erode(img))
#
#
# def soft_skeletonize(img, iter_num=50):
#     img1 = img.clone()
#     skel = torch.zeros_like(img)
#     for _ in range(iter_num):
#         opened = soft_open(img1)
#         tmp = torch.relu(img1 - opened)
#         skel = torch.relu(skel + tmp)
#         img1 = soft_erode(img1)
#     return torch.clamp(skel, 0, 1)

# Deepseek写法
# 采用了完整的腐蚀和膨胀操作，骨架更准确
# 加入了收敛停止条件
# 迭代次数少，3-10次按需调整
def soft_erode(img):
    """完整3x3腐蚀操作"""
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def soft_dilate(img):
    """完整3x3膨胀操作"""
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    """开运算：先腐蚀后膨胀"""
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iter_num=5):
    """
    改进的骨架化算法：
    1. 使用完整3x3邻域
    2. 添加收敛检查
    3. 减少迭代次数
    """
    skeleton = torch.zeros_like(img)
    for _ in range(iter_num):
        opened = soft_open(img)
        # 计算当前骨架层 = 原图 - 开运算结果
        sub = torch.clamp(img - opened, 0, 1)
        # 添加到总骨架
        skeleton = torch.clamp(skeleton + sub, 0, 1)
        # 更新图像 = 腐蚀结果
        img = soft_erode(img)

        # 收敛检查：当图像全黑时停止
        if torch.max(img) <= 0:
            break

    return skeleton


def compute_sdf_torch(gt: torch.Tensor) -> torch.Tensor:
    # 确保二值输入
    posmask = (gt > 0.5).bool()
    negmask = ~posmask

    # 创建有效mask (排除全前景/全背景)
    # spatial_dims = tuple(range(2, len(gt.shape)))
    # has_pos = torch.any(posmask, dim=spatial_dims)
    # has_neg = torch.any(negmask, dim=spatial_dims)
    has_pos = torch.any(posmask.flatten(start_dim=2), dim=2)  # (B, C)
    has_neg = torch.any(negmask.flatten(start_dim=2), dim=2)  # (B, C)
    valid_mask = has_pos & has_neg
    # 变成 N,C,1,1 方便广播
    valid_mask = valid_mask[:, :, None, None]

    # 初始化输出
    sdf = torch.zeros_like(gt, dtype=torch.float32)

    # 批量处理有效mask
    if torch.any(valid_mask):
        # valid_gt = gt[valid_mask]
        # valid_pos = posmask[valid_mask]
        # valid_neg = negmask[valid_mask]
        valid_gt = gt * valid_mask
        valid_pos = posmask * valid_mask
        valid_neg = negmask * valid_mask

        # 计算距离变换
        # pos_dis = kornia.morphology.distance_transform(valid_pos)
        # neg_dis = kornia.morphology.distance_transform(valid_neg)
        pos_dis = kornia.contrib.DistanceTransform()(valid_pos.float())
        neg_dis = kornia.contrib.DistanceTransform()(valid_neg.float())
        # clip，避免 inf / nan
        pos_dis = torch.clamp(pos_dis, max=1e6)  # 限制最大值
        neg_dis = torch.clamp(neg_dis, max=1e6)

        # sdf[valid_mask] = neg_dis - pos_dis
        # sdf = torch.where(valid_mask.expand_as(sdf), neg_dis - pos_dis, sdf)
        sdf = torch.where(valid_mask.expand_as(sdf), pos_dis - neg_dis, sdf)
        sdf = torch.clamp(sdf, min=-1e6, max=1e6)  # 限制范围
    # 处理无效情况
    # sdf[~valid_mask & posmask] = -1e6  # 全前景
    # sdf[~valid_mask & negmask] = 1e6  # 全背景
    if torch.isnan(sdf).any() or torch.isinf(sdf).any():
        print("⚠ NaN/Inf detected in SDF!")
        print("GT sum:", gt.sum().item())
    return sdf
