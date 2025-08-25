def compute_sdf_torch(self, gt: torch.Tensor) -> torch.Tensor:
    # 确保二值输入
    posmask = (gt > 0.5).bool()
    negmask = ~posmask

    # 创建有效mask (排除全前景/全背景)
    spatial_dims = tuple(range(2, len(gt.shape)))
    has_pos = torch.any(posmask, dim=spatial_dims)
    has_neg = torch.any(negmask, dim=spatial_dims)
    valid_mask = has_pos & has_neg

    # 初始化输出
    sdf = torch.zeros_like(gt, dtype=torch.float32)

    # 批量处理有效mask
    if torch.any(valid_mask):
        valid_gt = gt[valid_mask]
        valid_pos = posmask[valid_mask]
        valid_neg = negmask[valid_mask]

        # 计算距离变换
        pos_dis = kornia.morphology.distance_transform(valid_pos)
        neg_dis = kornia.morphology.distance_transform(valid_neg)

        sdf[valid_mask] = neg_dis - pos_dis

    # 处理无效情况
    sdf[~valid_mask & posmask] = -1e6  # 全前景
    sdf[~valid_mask & negmask] = 1e6  # 全背景

    return sdf