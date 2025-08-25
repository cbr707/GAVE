from pathlib import Path
import argparse
import json
import numpy as np
import torch
from torchvision import utils as vutils
from skimage import io
import os

from Tool.transformation import to_torch_tensors, pad_images_unet
from Hub.tool import factory
from types import SimpleNamespace

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from Hub.trainerNew import safe_load_weights

# 如果需要参数形式的代码，取消掉这段注释，并把 2 4 行 的args注释掉
# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--train_path', type=str, required=True)
# parser.add_argument('-i', '--images_path', type=str, required=True)
# parser.add_argument('-m', '--masks_path', type=str)
# parser.add_argument('-t', '--test_name', type=str, default="0716test")
# parser.add_argument('-g', '--gpu_id', type=int, default=0)
# args = parser.parse_args()
alpha = 1  # 控制概率图的截断位置
# 参数设置
args = SimpleNamespace(
    # 你的训练权重路径
    train_path=
    '''/data/disk8T2/caobr/202507GAVE/GAVE-main/Log/GAVE_test20250817-234632_SMPGAVENetV3_lr1e-04_RRLoss-BTCB3Loss_bc16/''',
    # images_path='../Data/GAVE/validation/images',  # 验证图像路径
    # masks_path='../Data/GAVE/validation/masks',      # 掩码路径
    # images_path='/data/disk8T2/lincy/Lincy2025/GAVE-main/Data/GAVE_enhanced/validation_enhanced/images',  # 验证图像路径
    # masks_path='/data/disk8T2/lincy/Lincy2025/GAVE-main/Data/GAVE_enhanced/validation_enhanced/masks',      # 掩码路径
    # images_path='/home/caobr/caobr/202507GAVE/GAVE-main/Data-enhanced/GAVE/validation_enhanced/images',  # 验证图像路径
    # masks_path='/data/disk8T2/caobr/202507GAVE/GAVE-main/Data-enhanced/GAVE/validation_enhanced/masks',      # 掩码路径
    # images_path='/home/caobr/caobr/202507GAVE/GAVE-main/Data-enhanced/GAVE/training/images',  # 验证图像路径
    # masks_path='/data/disk8T2/caobr/202507GAVE/GAVE-main/Data-enhanced/GAVE/training/masks',  # 掩码路径
    images_path='/home/caobr/caobr/202507GAVE/GAVE-main/Data-enhanced/GAVE/41-50/images',  # 验证图像路径
    masks_path='/data/disk8T2/caobr/202507GAVE/GAVE-main/Data-enhanced/GAVE/41-50/masks',  # 掩码路径
    alpha=alpha,  # 控制概率图的截断位置
    test_name='val2',  # 本次预测任务名称
    gpu_id=0,  # 使用的GPU编号
    temperature=1,  # 控制概率图的高低，小于1会提升整体的预测概率
    logit_bias=0,  # 控制概率图的高低，正数会提升整体的预测概率

)


def save_probability_map(logits, temperature, logit_bias=0.0, clip=True, mask_threshold=10):
    """
    logits: (B,1,H,W) 或 (B,H,W) 的原始模型输出（未过 sigmoid）
    temperature: 温度缩放（T<1 → 概率更“极端”，T>1 → 更“温和”）
    logit_bias:  正值整体抬高血管概率，负值整体压低
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)  # -> (B,1,H,W)
        # 创建 mask：logits > mask_threshold 的像素才做温度缩放/加 bias
    # foreground_mask = logits > mask_threshold
    #
    # # 初始化调整后的 logits
    # z = logits.clone()

    # 仅对前景区域调整
    # z[foreground_mask] = logits[foreground_mask] / max(temperature, 1e-6) + logit_bias
    z = logits / max(temperature, 1e-6) + logit_bias
    prob = z

    # sigmoid 转概率
    # prob = torch.sigmoid(z)

    # clip 到 [0,1]
    # if clip:
    #     prob = prob.clamp(0.0, 1.0)

    # 保存图像
    # vutils.save_image(prob, fname)

    return prob


print('Loading model')
checkpoint = torch.load(Path(args.train_path, 'generator_best.pth'))

print('Loading config')
with open(Path(args.train_path, 'config.json'), 'r') as f:
    config = json.load(f)

print('Config:')
print(json.dumps(config, indent=4))

config = argparse.Namespace(**config)

print('Creating model')
model = factory.ModelFactory().create_class(
    config.model,
    config.in_channels,
    config.out_channels,
    config.base_channels,
    config.num_iterations,
    config.encoder_name,
    config.encoder_weights,
    config.encoder_depth,

)

print('Loading weights')
model.load_state_dict(checkpoint)

if torch.cuda.is_available():
    model.cuda(args.gpu_id)

model.eval()

images_path = Path(args.images_path)
masks_path = Path(args.masks_path)
assert images_path.exists(), images_path
assert masks_path.exists(), masks_path

save_path = Path(args.train_path) / 'pred'
if args.test_name is not None:
    save_path = save_path / args.test_name
save_path.mkdir(exist_ok=True, parents=True)

for image_fn in sorted(images_path.iterdir()):
    mask_fn = None
    for mask_fn in masks_path.iterdir():
        if mask_fn.stem == image_fn.stem:
            break
    if mask_fn is None:
        print(f'ERROR: Mask not found for {image_fn.name}')
        exit(1)
    if image_fn.is_file():
        print(f'> Processing {image_fn.name}')
        image = io.imread(image_fn)
        if image.max() > 1:
            image = (image / 255.0)[..., :3]
        mask = io.imread(mask_fn).astype(np.float32)
        if mask.max() > 1:
            mask = mask / 255.0
        images, paddings = pad_images_unet([image, mask], return_paddings=True)
        img = images[0]
        padding = paddings[0]
        mask = images[1]
        mask = np.stack([mask, ] * 3, axis=2)
        mask_padding = paddings[1]
        tensors = to_torch_tensors([img, mask])
        tensor = tensors[0]
        mask_tensor = tensors[1]
        if torch.cuda.is_available():
            tensor = tensor.cuda(args.gpu_id)
        else:
            tensor = tensor.cpu()
        tensor = tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        with torch.no_grad():
            preds = model(tensor)
            if isinstance(preds, list):
                pred = preds[-1]
            else:
                pred = preds
            # 兼容性代码 单双通道结果保护
            if pred.shape[1] == 1:
                mask_tensor = mask_tensor[:, 0:1, :, :]
            elif pred.shape[1] == 2:
                mask_tensor = mask_tensor[:, 0:2, :, :]

            # save_probability_map(pred, args.temperature, logit_bias=args.logit_bias)
            pred = torch.sigmoid(pred)
            pred[mask_tensor < 0.5] = 0
            pred1 = pred

            # 批处理模块
            # for alpha in range(0, 16):
            #     alpha = round((1.3 ** alpha)*50)
            #     pred = pred1 * alpha
            #     pred = pred.clamp(min=0, max=1)
            #     pred = pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            #     target_fn = save_path / str(alpha)
            #     target_fn.mkdir(exist_ok=True, parents=True)
            #     target_fn = target_fn / Path(image_fn).name
            #     vutils.save_image(pred, target_fn)

            if alpha!=1:
                pred = pred*args.alpha
                pred = pred.clamp(min=0, max=1)
            pred = pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            target_fn = save_path / Path(image_fn).name
            vutils.save_image(pred, target_fn)

print('Images saved in', save_path)
