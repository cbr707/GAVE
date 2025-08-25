import torch.nn as nn
import torch
import torchvision.utils as vutils
import tifffile
from Loss.clDiceLoss import *
import scipy.ndimage as ndi


def sigmoid_gate(p, tau=0.5, k=30):
    # p 已经是 sigmoid 后的概率
    return torch.sigmoid(k * (p - tau))


# 基础3类BCEloss
class BCE3Loss(nn.Module):
    """
        A = 0+1, Vessel = 0+1+2, V = 1+2
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vessels, vessels, mask):
        mask = mask[:, 0, :, :]
        mask = torch.round(mask)

        pred_a = pred_vessels[:, 0, :, :]
        pred_vt = pred_vessels[:, 1, :, :]
        pred_v = pred_vessels[:, 2, :, :]

        gt_a = vessels[:, 0, :, :] + vessels[:, 1, :, :]  # A = 0 + 1
        gt_vt = vessels[:, 0, :, :] + vessels[:, 1, :, :] + vessels[:, 2, :, :]  # Vessel tree = 0 + 1 + 2
        gt_v = vessels[:, 1, :, :] + vessels[:, 2, :, :]  # V = 1 + 2

        gt_a = torch.clamp(gt_a, 0, 1)
        gt_vt = torch.clamp(gt_vt, 0, 1)
        gt_v = torch.clamp(gt_v, 0, 1)

        loss = self.loss(pred_a[mask > 0.], gt_a[mask > 0.])
        loss += self.loss(pred_vt[mask > 0.], gt_vt[mask > 0.])
        loss += self.loss(pred_v[mask > 0.], gt_v[mask > 0.])

        return loss

    def save_predicted(self, prediction, fname):
        prediction_processed = self.process_predicted(prediction)
        vutils.save_image(prediction_processed, fname)

    def process_predicted(self, prediction):
        if isinstance(prediction, list):
            prediction = prediction[-1]

            # 在两个通道之间插入一个空通道
        if prediction.shape[1] == 2:  # 只在 2 通道时插入
            B, _, H, W = prediction.shape
            empty_channel = torch.zeros(
                (B, 1, H, W),
                device=prediction.device,
                dtype=prediction.dtype
            )
            prediction = torch.cat([prediction[:, :1], empty_channel, prediction[:, 1:]], dim=1)

            return prediction
        return torch.sigmoid(prediction.clone())


# a_only+vt+v_only+a*v特殊4类，一般不用
class BCE4Loss(BCE3Loss):
    """
            直接对四个通道分别计算 BCE loss：
            0: a
            1: bv
            2: v
            3: av
        """

    def __init__(self):
        super().__init__()  # 继承 BCE3Loss 的 save_predicted / process_predicted
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vessels, vessels, mask):
        mask = mask[:, 0, :, :]
        mask = torch.round(mask)

        pred_a = pred_vessels[:, 0, :, :]
        pred_vt = pred_vessels[:, 1, :, :]
        pred_v = pred_vessels[:, 2, :, :]
        pred_av = pred_vessels[:, 3, :, :]

        gt_a = vessels[:, 0, :, :]  # A = 0
        gt_vt = vessels[:, 0, :, :] + vessels[:, 1, :, :] + vessels[:, 2, :, :]  # Vessel tree = 0 + 1 + 2
        gt_v = vessels[:, 2, :, :]  # V = 2
        gt_av = vessels[:, 1, :, :]  # av = 1

        gt_a = torch.clamp(gt_a, 0, 1)
        gt_vt = torch.clamp(gt_vt, 0, 1)
        gt_v = torch.clamp(gt_v, 0, 1)
        gt_av = torch.clamp(gt_av, 0, 1)

        loss = self.loss(pred_a[mask > 0.], gt_a[mask > 0.])
        loss += self.loss(pred_vt[mask > 0.], gt_vt[mask > 0.])
        loss += self.loss(pred_v[mask > 0.], gt_v[mask > 0.])
        loss += self.loss(pred_av[mask > 0.], gt_av[mask > 0.])

        return loss

    def save_predicted(self, prediction, fname):
        # 处理成概率
        prediction_processed = self.process_predicted(prediction)  # (1,4,H,W)
        prediction_processed = prediction_processed.squeeze(0)  # (4,H,W)

        # 保存 4 张单通道 PNG
        for ch in range(prediction_processed.size(0)):
            ch_img = prediction_processed[ch:ch + 1, :, :]  # (1,H,W)
            save_name = fname.replace('.png', f'_ch{ch}.png')
            vutils.save_image(ch_img, save_name, normalize=True)

        # 保存 4 通道 tiff (float32)
        tiff_name = fname.replace('.png', '.tiff')
        tiff_data = prediction_processed.cpu().numpy().astype('float32')  # (4,H,W)
        tifffile.imwrite(tiff_name, tiff_data)


# 基础3类外加交叉点
class BCE4LossNew(BCE4Loss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_vessels, vessels, mask):
        mask = mask[:, 0, :, :]
        mask = torch.round(mask)

        pred_a = pred_vessels[:, 0, :, :]
        pred_vt = pred_vessels[:, 1, :, :]
        pred_v = pred_vessels[:, 2, :, :]
        pred_av = pred_vessels[:, 3, :, :]

        gt_a = vessels[:, 0, :, :] + vessels[:, 1, :, :]  # A = 0 + 1
        gt_vt = vessels[:, 0, :, :] + vessels[:, 1, :, :] + vessels[:, 2, :, :]  # Vessel tree = 0 + 1 + 2
        gt_v = vessels[:, 1, :, :] + vessels[:, 2, :, :]  # V = 1 + 2
        gt_av = vessels[:, 1, :, :]  # av = 1

        gt_a = torch.clamp(gt_a, 0, 1)
        gt_vt = torch.clamp(gt_vt, 0, 1)
        gt_v = torch.clamp(gt_v, 0, 1)
        gt_av = torch.clamp(gt_av, 0, 1)

        loss = self.loss(pred_a[mask > 0.], gt_a[mask > 0.])
        loss += self.loss(pred_vt[mask > 0.], gt_vt[mask > 0.])
        loss += self.loss(pred_v[mask > 0.], gt_v[mask > 0.])
        loss += self.loss(pred_av[mask > 0.], gt_av[mask > 0.])

        return loss


class RRLoss(nn.Module):
    """
    Recursive refinement loss.
    """

    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, predictions, gt, mask):
        loss_1 = self.base_criterion(predictions[0], gt, mask)
        if len(predictions) == 1:
            return loss_1

        # Second loss (refinement) inspired by Mosinska:CVPR:2018.
        loss_2 = 1 * self.base_criterion(predictions[1], gt, mask)
        if len(predictions) == 2:
            return loss_1 + loss_2
        for i, prediction in enumerate(predictions[2:], 2):
            loss_2 += i * self.base_criterion(prediction, gt, mask)

        K = len(predictions[1:])
        Z = (1 / 2) * K * (K + 1)

        loss_2 *= 1 / Z

        loss = loss_1 + loss_2

        return loss

    def save_predicted(self, predictions, fname):
        self.base_criterion.save_predicted(predictions[-1], fname)

    def process_predicted(self, predictions):
        new_predictions = []
        for prediction in predictions:
            new_predictions.append(self.base_criterion.process_predicted(prediction))
        return new_predictions


class RRLossNew(RRLoss):
    def __init__(self, base_criterion):
        super().__init__(base_criterion)

    def forward(self, predictions, gt, mask):
        # 初始化第二层 loss
        loss_2 = 1 * self.base_criterion(predictions[0], gt, mask)

        # 如果有更多循环 refinement 层
        for i, prediction in enumerate(predictions[1:], 1):  # 序号从1开始加权
            loss_2 += (i + 1) * self.base_criterion(prediction, gt, mask)

        # 归一化
        K = len(predictions)
        Z = 0.5 * K * (K + 1)
        loss_2 *= 1 / Z

        return loss_2


class BCEWithTverskyLoss(nn.Module):
    """
    BCE + Tversky Loss
    Tversky 公式:
        Tversky = TP / (TP + α * FN + β * FP)
    当 α=β=0.5 时等价于 Dice 系数
    参数:
        bce_weight: BCE 部分权重
        tversky_weight: Tversky 部分权重
        alpha: FN 惩罚系数（越大越关注 recall）
        beta: FP 惩罚系数（越大越关注 precision）
        smooth: 平滑项，防止分母为 0
    """

    def __init__(self, bce_weight=0.4, tversky_weight=2, alpha=0.9, beta=0.1, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        pred: logits, shape = (B, 1, H, W) 或 (B, H, W)
        target: 0/1 tensor, shape 同 pred
        """
        # 保证 shape 一致
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        # BCE 部分
        bce = self.bce_loss(pred, target)

        # Tversky 部分
        pred_prob = torch.sigmoid(pred)
        tp = (pred_prob * target).sum()
        fn = ((1 - pred_prob) * target).sum()
        fp = (pred_prob * (1 - target)).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1 - tversky_index

        # 总损失
        return self.bce_weight * bce + self.tversky_weight * tversky_loss


# 暂时不用
class BCETverskyFocalLoss(nn.Module):
    def __init__(self,
                 bce_weight=0.35,
                 tversky_weight=0.35,
                 focal_weight=0.30,  # [MOD] 新增：Focal loss 权重
                 alpha=0.3, beta=0.7,  # Tversky：β>α 强调 FN 惩罚
                 gamma_focal=0.75,  # [MOD] 新增：Focal 的 γ（0.5~1.5 常用）
                 alpha_focal=0.75,  # [MOD] 新增：Focal 的 α（正类权重，血管稀疏→>0.5）
                 smooth=1e-6,
                 bce_temperature=1.0):  # [MOD] 新增：BCE 的温度缩放（T<=1 更“激进”）
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight  # [MOD]
        self.alpha = alpha
        self.beta = beta
        self.gamma_focal = gamma_focal  # [MOD]
        self.alpha_focal = alpha_focal  # [MOD]
        self.smooth = smooth
        self.bce_temperature = bce_temperature  # [MOD]
        self._bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred, target):
        """
        pred: logits, shape = (B, 1, H, W) 或 (B, H, W)
        target: 0/1 tensor, shape 同 pred
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        # ---------------------------
        # BCE（带温度缩放）
        # ---------------------------
        # [MOD] 对 logits 做温度缩放：T<1 → 更“锐化”，T>1 → 更“柔和”
        if self.bce_temperature != 1.0:
            pred_bce = pred / self.bce_temperature
        else:
            pred_bce = pred

        bce = self._bce(pred_bce, target)

        # ---------------------------
        # Tversky
        # ---------------------------
        pred_prob = torch.sigmoid(pred)  # 注意：Tversky 和 Focal 都在概率上计算
        tp = (pred_prob * target).sum()
        fn = ((1.0 - pred_prob) * target).sum()
        fp = (pred_prob * (1.0 - target)).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1.0 - tversky_index

        # ---------------------------
        # [MOD] Focal Loss（二分类，概率版）
        # ---------------------------
        # p_t = p     if y=1
        #       1-p   if y=0
        p = pred_prob
        y = target
        eps = 1e-8
        p_t = p * y + (1.0 - p) * (1.0 - y)
        # α_t：正类用 alpha_focal，负类用 1-alpha_focal
        alpha_t = self.alpha_focal * y + (1.0 - self.alpha_focal) * (1.0 - y)
        focal = - alpha_t * torch.pow(1.0 - p_t, self.gamma_focal) * torch.log(p_t.clamp(min=eps))
        focal_loss = focal.mean()

        # ---------------------------
        # 总损失
        # ---------------------------
        loss = self.bce_weight * bce + self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
        return loss


class ClDiceLoss(nn.Module):
    """
    clDice loss for topology preservation in thin structures segmentation.
    """

    def __init__(self, iter_num=5, smooth=1.0, theta=7):
        super(ClDiceLoss, self).__init__()
        self.iter_num = iter_num
        self.smooth = smooth
        self.theta = theta

    def forward(self, pred, target):
        """
        pred:   [B, 1, H, W] - probability map (without sigmoid)
        target: [B, 1, H, W] - binary ground truth
        """
        # 兼容性模块
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred = torch.sigmoid(pred)
        if self.theta != 1:
            pred = pred ** self.theta
        # pred = sigmoid_gate(pred, tau=0.5, k=30)  # 近似阶跃处理
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        pred_skel = soft_skeletonize(pred, self.iter_num)
        target_skel = soft_skeletonize(target, self.iter_num)

        # tprec = (torch.sum(pred_skel * target) + self.smooth) / \
        #         (torch.sum(pred_skel) + self.smooth)
        # tsens = (torch.sum(target_skel * pred) + self.smooth) / \
        #         (torch.sum(target_skel) + self.smooth)
        #
        # cl_dice = (2 * tprec * tsens) / (tprec + tsens)
        # return 1 - cl_dice  # loss

        # ========= 加权部分 =========
        # target: [B,1,H,W] 二值血管图
        # kernel = torch.ones(1, 1, 7, 7, device=target.device)  # 5x5卷积邻域，可调整
        # nbr = F.conv2d(target, kernel, padding=3)  # 邻域内血管像素数量
        # 池化方法，保持 nbr 大小一致
        pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3, count_include_pad=False)
        nbr = pool(target) * (7 * 7)  # 还原邻域内像素数量

        # 粗血管掩码：邻域内血管像素多于阈值
        coarse_mask = (nbr >= 25).float()  # 阈值可根据分辨率调整
        fine_mask = target - coarse_mask
        fine_mask = torch.clamp(fine_mask, 0, 1)

        # 膨胀操作（扩大3个像素），关键覆盖连接处
        # coarse_mask = dilate_mask(coarse_mask, dilation=3)
        # fine_mask = dilate_mask(fine_mask, dilation=3)
        dilate3 = nn.MaxPool2d(
            kernel_size=7,  # 3x3 膨胀
            stride=1,
            padding=3  # 保持尺寸不变
        )

        coarse_mask = dilate3(coarse_mask)
        fine_mask = dilate3(fine_mask)

        # gt_skel 二值化
        gt_skel_bin = (target_skel > 0.5).float()
        kernel = torch.ones(1, 1, 3, 3, device=gt_skel_bin.device)
        nbr = torch.nn.functional.conv2d(gt_skel_bin, kernel, padding=1)

        end_mask = ((nbr == 2) & (gt_skel_bin > 0.5)).float()  # 度=1
        junction_mask = ((nbr >= 4) & (gt_skel_bin > 0.5)).float()  # 度≥3

        w = 5 + 1 * end_mask + 50 * junction_mask  # 端点×2，交叉×3

        # ========= 加权 clDice =========
        # inter = (pred_skel * target * w).sum()
        # pred_len = (pred_skel * w).sum().clamp_min(1.0)
        # gt_len = (target_skel * w).sum().clamp_min(1.0)
        # 加权 clDice 新写法
        inter = (pred_skel * fine_mask * w).sum() + (pred_skel * coarse_mask).sum()
        pred_len = (pred_skel * (fine_mask * w + coarse_mask)).sum().clamp_min(1.0)
        gt_len = (fine_mask * w + coarse_mask).sum().clamp_min(1.0)

        tprec = (inter + self.smooth) / (pred_len + self.smooth)
        tsens = (inter + self.smooth) / (gt_len + self.smooth)

        cl_dice = (2 * tprec * tsens) / (tprec + tsens + 1e-6)
        return 1 - cl_dice  # loss



# 未测试
class SoftCenterlineLoss(nn.Module):
    def __init__(self,
                 mode='exp',  # 'exp' 或 'pow' 两种尖化方式
                 sigma_ratio=1.0,  # [MOD] 自适应 σ：σ = sigma_ratio * 正类D的均值（避免手工调绝对σ）
                 pow_gamma=2.0,  # [MOD] 当 mode='pow' 时的指数，>1 更尖锐
                 use_bce=True,  # True: BCE；False: MSE
                 loss_weight=1.0):
        super().__init__()
        self.mode = mode
        self.sigma_ratio = sigma_ratio
        self.pow_gamma = pow_gamma
        self.use_bce = use_bce
        self.loss_weight = loss_weight
        self._bce = nn.BCELoss(reduction='mean')
        self._mse = nn.MSELoss(reduction='mean')

    @torch.no_grad()
    def _soft_centerline_target(self, target):
        """
        target: (B,1,H,W) 0/1 tensor
        return C: (B,1,H,W) in [0,1]
        """
        # 距离变换：前景内到边界的距离（0 在边界，中心线最大）
        # kornia 接受形如 (B,1,H,W) 的 bool/float tensor
        tgt = (target > 0.5).float()
        # 防止全零报错
        if tgt.sum() == 0:
            return torch.zeros_like(tgt)

        # Kornia 的 distance_transform 默认计算到零像素的距离（这里前景=1，所以先取前景）
        # 只在前景内保留距离，背景为 0
        # 下面两步：对前景和背景的距离变换都算出来，再把前景内的距离拿出来
        pos_dis = kornia.morphology.distance_transform(tgt)  # 前景像素到最近背景的距离
        # 归一化（自适应）：按每张图正类的 D 分布设置 σ 或 max
        B = tgt.shape[0]
        C_list = []
        eps = 1e-6
        for b in range(B):
            d = pos_dis[b:b + 1]  # (1,1,H,W)
            mask = tgt[b:b + 1] > 0.5
            if mask.sum() == 0:
                C_list.append(torch.zeros_like(d))
                continue
            d_pos = d[mask]
            d_norm = d / (d_pos.max() + eps)  # [0,1] 自适应到该样本的前景宽度
            if self.mode == 'exp':
                sigma = max(self.sigma_ratio * d_pos.mean().item(), 0.5)  # [MOD] σ 自适应，>=0.5 防止过尖
                C = torch.exp(- torch.square(d / (sigma + eps)))
            else:  # 'pow'
                C = torch.pow(d_norm.clamp(0, 1), self.pow_gamma)
            # 只保留前景内的中心线热力图，背景为 0
            C = C * mask.float()
            C_list.append(C)

        C = torch.cat(C_list, dim=0)
        return C.clamp(0.0, 1.0)

    def forward(self, pred, target):
        """
        pred: logits, (B,1,H,W)
        target: 0/1,  (B,1,H,W)
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        with torch.no_grad():
            C = self._soft_centerline_target(target)  # (B,1,H,W) in [0,1]

        pred_prob = torch.sigmoid(pred)
        if self.use_bce:
            loss_c = self._bce(pred_prob, C)  # [MOD] 软中心线监督（BCE）
        else:
            loss_c = self._mse(pred_prob, C)  # [MOD] 软中心线监督（MSE）

        return self.loss_weight * loss_c


class BCETverskyClDice(nn.Module):
    def __init__(self, bce_tversky_loss, cldice_loss, alpha=1):
        super().__init__()
        self.bce_tversky_loss = bce_tversky_loss
        self.cldice_loss = cldice_loss
        self.alpha = alpha

    def forward(self, pred, target):
        """
                pred:   [B, 1, H, W] - probability map (without sigmoid)
                target: [B, 1, H, W] - binary ground truth
                """
        return self.bce_tversky_loss(pred, target) + \
            self.alpha * self.cldice_loss(pred, target)
        # return self.bce_tversky_loss(pred, target)


# 只加在粗血管及其边界
class BoundaryLoss(nn.Module):
    def __init__(self, epsilon=1e-6, weight_pos=1, neighbor_kernel=7, neighbor_threshold=25, dilate_radius=3):
        super().__init__()
        self.epsilon = epsilon  # 避免除零
        self.weight_pos = weight_pos
        # ========= 新增参数：粗血管提取相关 =========
        self.neighbor_kernel = neighbor_kernel  # 邻域卷积核大小
        self.neighbor_threshold = neighbor_threshold  # 邻域内像素阈值判定粗血管
        self.dilate_radius = dilate_radius  # 粗血管膨胀半径

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, 3, H, W] - probability map (without sigmoid)
        gt: [B, 3, H, W] - binary ground truth
        """
        pred = torch.sigmoid(pred).float()
        gt = gt.float()
        # ================= 新增：粗血管mask提取 =================
        # 1. 对每个通道进行邻域统计
        B, C, H, W = gt.shape
        mask_coarse = torch.zeros_like(gt)
        for c in range(C):
            gt_c = gt[:, c:c + 1, :, :]  # [B,1,H,W]
            # kernel = torch.ones(1, 1, self.neighbor_kernel, self.neighbor_kernel, device=gt.device)
            # nbr = F.conv2d(gt_c, kernel, padding=self.neighbor_kernel // 2)
            # AvgPool 替代 Conv2d
            pool = nn.AvgPool2d(
                kernel_size=self.neighbor_kernel,
                stride=1,
                padding=self.neighbor_kernel // 2,
                count_include_pad=False
            )

            # 池化结果是均值，乘以邻域大小恢复成总和
            nbr = pool(gt_c) * (self.neighbor_kernel * self.neighbor_kernel)
            # 2. 粗血管mask
            coarse = ((nbr > self.neighbor_threshold) & (gt_c > 0.5)).float()
            # 3. 膨胀粗血管mask
            dilate_kernel_size = 2 * self.dilate_radius + 1
            # dilate_kernel = torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, device=gt.device)
            # coarse_dilated = (F.conv2d(coarse, dilate_kernel, padding=self.dilate_radius) > 0).float()
            # 使用 MaxPool2d 实现膨胀
            dilate = nn.MaxPool2d(
                kernel_size=dilate_kernel_size,
                stride=1,
                padding=self.dilate_radius
            )

            coarse_dilated = dilate(coarse)
            mask_coarse[:, c:c + 1, :, :] = coarse_dilated
        # ================= 结束新增 =================

        sdf = compute_sdf_torch(gt)

        # 前景/背景 mask
        pos_mask = (gt > 0.5).float()
        neg_mask = 1 - pos_mask

        # ================ 修改加权部分，限制在粗血管mask =================
        # 原 weight_map = self.weight_pos * pos_mask + neg_mask
        weight_map = mask_coarse * (self.weight_pos * pos_mask + neg_mask)
        # =================================================================

        # 归一化处理
        norm_factor = (gt.shape[-1] * gt.shape[-2]) ** 0.5 + self.epsilon
        sdf_normalized = sdf / norm_factor

        return torch.mean(sdf_normalized * pred * weight_map)


# 用于VT单通道全血管
class BCETverskyLossVT(BCE3Loss):
    """
    单分类版本的 BCE + Tversky Loss
    Tversky 公式:
        Tversky = TP / (TP + α * FN + β * FP)
    当 α=β=0.5 时等价于 Dice 系数
    参数:
        bce_weight: BCE 部分权重
        tversky_weight: Tversky 部分权重
        alpha: FN 惩罚系数（越大越关注 recall）
        beta: FP 惩罚系数（越大越关注 precision）
        偏向 recall（减少漏检）：alpha > beta，例如 alpha=0.7, beta=0.3
    """

    def __init__(self, **kwargs):
        super().__init__()
        # 切换Loss
        # self.core_loss = BCEWithTverskyLoss(**kwargs)
        self.core_loss = BCETverskyClDice(BCEWithTverskyLoss(), ClDiceLoss())
        # self.core_loss = BCETverskyClDice(BCETverskyFocalLoss(), ClDiceLoss())

    def _to_single_channel(self, target):
        """
        将多通道 target 转为单通道 (默认: 前三通道求和并 clamp 到 0/1)
        其他 Loss 只需要重写这个方法即可更换通道处理逻辑
        """
        if target.size(1) != 1:
            target = target[:, 0:3, :, :].sum(dim=1, keepdim=True).clamp(0, 1)
        return target

    def _to_single_pred(self, pred):
        return pred[:, 1, :, :]

    def forward(self, pred, target, mask=None):
        """
                            pred:   [B, 3, H, W] - probability map (without sigmoid)
                            target: [B, 3, H, W] - binary ground truth
                            """
        # 调用可重写的通道处理方法
        target = self._to_single_channel(target)

        if isinstance(pred, list):
            pred = pred[-1]
        if mask is not None:
            mask = torch.round(mask[:, 0, :, :])
            # 展平到1维
            # pred = self._to_single_pred(pred)[mask > 0]
            # target = target[:, 0, :, :][mask > 0]
            # 保持三维
            pred = self._to_single_pred(pred)
            pred = pred * (mask > 0).float()
            target = target[:, 0, :, :]
            target = target * (mask > 0).float()
        else:
            pred = self._to_single_pred(pred)
            target = target[:, 0, :, :]
        return self.core_loss(pred, target)


# 用于A动脉
class BCETverskyLossA(BCETverskyLossVT):
    def _to_single_channel(self, target):
        """
        自定义的多通道转单通道逻辑
        这里改成只取第 1 通道（索引 0）
        """
        if target.size(1) != 1:
            target = target[:, 0:2, :, :].sum(dim=1, keepdim=True).clamp(0, 1)  # 只取0,1通道
        return target

    def _to_single_pred(self, pred):
        return pred[:, 0, :, :]


# 用于V静脉
class BCETverskyLossV(BCETverskyLossVT):
    def _to_single_channel(self, target):
        """
        自定义的多通道转单通道逻辑
        这里改成只取第 1 通道（索引 0）
        """
        if target.size(1) != 1:
            target = target[:, 1:3, :, :].sum(dim=1, keepdim=True).clamp(0, 1)  # 只取1，2通道
        return target

    def _to_single_pred(self, pred):
        return pred[:, 2, :, :]


# 用于A*V交叉点
class BCETverskyLossAV(BCETverskyLossVT):
    def _to_single_channel(self, target):
        """
        自定义的多通道转单通道逻辑
        这里改成只取第 1 通道（索引 0）
        """
        if target.size(1) != 1:
            target = target[:, 1, :, :]  # 只取1通道
        return target

    def _to_single_pred(self, pred):
        return pred[:, 3, :, :]


class BCETversky2Loss(BCE3Loss):
    """
    将同一预测值/GT 送入 2 个 BCETverskyLoss 子类，loss 加和
    适用于多通道预测，每个子 loss 自己定义取通道逻辑
    """

    def __init__(self):
        """
        loss_a / loss_bv / loss_v :
            都是继承自 BCETverskyLoss 的类实例，
            内部已实现 _to_single_channel() 决定取哪一通道
        """
        super().__init__()
        # self.core_loss = BCETverskyClDice(BCEWithTverskyLoss(), ClDiceLoss())
        self.losses = nn.ModuleList([BCETverskyLossA(), BCETverskyLossV()])

    def forward(self, pred, target, mask=None):
        """
                            pred:   [B, 2, H, W] - probability map (without sigmoid)
                            target: [B, 3, H, W] - binary ground truth
                            """
        # 在两个通道之间插入一个全 0 通道,不参与运算
        B, _, H, W = pred.shape
        zero_channel = torch.zeros((B, 1, H, W), device=pred.device, dtype=pred.dtype)
        pred = torch.cat([pred[:, :1], zero_channel, pred[:, 1:]], dim=1)  # [B, 3, H, W]
        total_loss = 0.0
        for loss_fn in self.losses:
            total_loss += loss_fn(pred, target, mask)
        return total_loss


class BCETversky3Loss(BCE3Loss):
    """
    将同一预测值/GT 送入 3 个 BCETverskyLoss 子类，loss 加和
    适用于多通道预测，每个子 loss 自己定义取通道逻辑
    """

    def __init__(self, pseudo_weight=0.2, low=0.1, high=0.9):
        """
        loss_a / loss_bv / loss_v :
            都是继承自 BCETverskyLoss 的类实例，
            内部已实现 _to_single_channel() 决定取哪一通道
        """
        super().__init__()
        # self.core_loss = BCETverskyClDice(BCEWithTverskyLoss(), ClDiceLoss())
        self.losses = nn.ModuleList([BCETverskyLossVT(), BCETverskyLossA(), BCETverskyLossV()])

        # === 新增: 伪标签超参数 ===
        self.pseudo_weight = pseudo_weight  # 伪标签 loss 权重
        self.low = low  # 负类阈值
        self.high = high  # 正类阈值

    def forward(self, pred, target, mask=None):
        """
                            pred:   [B, 3, H, W] - probability map (without sigmoid)
                            target: [B, 3, H, W] - binary ground truth
                            """
        total_loss = 0.0
        for loss_fn in self.losses:
            total_loss += loss_fn(pred, target, mask)

        # # === 新增: 伪标签损失 (只在 mask 内计算) ===
        # if mask is not None:
        #     with torch.no_grad():
        #         probs = torch.sigmoid(pred)  # [B,3,H,W]
        #         pseudo = torch.full_like(probs, -1)  # -1 表示忽略
        #         pseudo[(probs >= self.high) & (mask > 0.5)] = 1
        #         # pseudo[(probs <= self.low) & (mask > 0.5)] = 0
        #
        #     if (pseudo != -1).any():  # 有有效伪标签时才计算
        #         pseudo_loss = F.binary_cross_entropy_with_logits(
        #             pred[pseudo != -1],
        #             pseudo[pseudo != -1].float()
        #         )
        #         total_loss += self.pseudo_weight * pseudo_loss
        return total_loss


# 先别用
class BTCB3Loss(BCE3Loss):
    def __init__(self, alpha=0.1):
        """
        loss_a / loss_bv / loss_v :
            都是继承自 BCETverskyLoss 的类实例，
            内部已实现 _to_single_channel() 决定取哪一通道
        """
        super().__init__()
        self.core_loss = BCETversky3Loss()
        self.boundary_loss = BoundaryLoss()
        self.alpha = alpha

    def forward(self, pred, target, mask=None):
        if isinstance(pred, list):
            pred = pred[-1]
        core_loss = self.core_loss(pred, target)

        # 构造三通道 GT
        gt_a = target[:, 0, :, :] + target[:, 1, :, :]  # A = 0 + 1
        gt_vt = target[:, 0, :, :] + target[:, 1, :, :] + target[:, 2, :, :]  # Vessel tree
        gt_v = target[:, 1, :, :] + target[:, 2, :, :]  # V = 1 + 2

        gt_a = torch.clamp(gt_a, 0, 1)
        gt_vt = torch.clamp(gt_vt, 0, 1)
        gt_v = torch.clamp(gt_v, 0, 1)

        target = torch.stack([gt_a, gt_vt, gt_v], dim=1)  # (B, 3, H, W)

        # pred 保持 (B, 3, H, W)

        if mask is not None:
            mask = torch.round(mask[:, 0, :, :])  # (B, H, W)
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
            pred = pred * mask
            target = target * mask

        return core_loss + self.alpha * self.boundary_loss(pred, target)  # 直接三通道一起算


class BCETversky4Loss(BCETversky3Loss):
    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList([BCETverskyLossVT(), BCETverskyLossA(), BCETverskyLossV(), BCETverskyLossAV()])
