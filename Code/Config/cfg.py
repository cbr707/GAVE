import argparse
import socket
from datetime import datetime


def str2int_tuple(v):
    return tuple(int(x) for x in v.split(','))


# 增加了根据时间命名
now_str = datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser()
# 指定GPU
parser.add_argument('--gpu_id', type=int, default=1)
# 指定训练数据，是否加载全血管分割结果，'GAVE+seg'或者GAVE
parser.add_argument('--dataset', type=str, default='GAVE')
# parser.add_argument('--dataset', type=str, default='GAVE+seg')

'''网络选择
 "SMPUnet" 'GAVENet' "RRSMPUNet"'GAVENetV2''UNet'
'''
# parser.add_argument('--model', type=str, default='UNet')
# parser.add_argument('--model', type=str, default='SMPUNet') # 3 in
# parser.add_argument('--model', type=str, default='SMPUNetV2') # 6 in
# parser.add_argument('--model', type=str, default='SMPUNetV3')  # 6 in 循环结构
# parser.add_argument('--model', type=str, default='GAVENet')
# parser.add_argument('--model', type=str, default='GAVENetV3')
parser.add_argument('--model', type=str, default='SMPGAVENetV3')  # 3 in 6 in 3 out 全图递归
# parser.add_argument('--model', type=str, default='SMPGAVENetV4')  # 3 in 3 in 3 out 递归血管
# parser.add_argument('--model', type=str, default='SMPGAVENet')
# 使用该网络请使用对应的训练文件
# parser.add_argument('--model', type=str, default='RRSMPUNet')

'''损失函数选择 '''
# 使用RRLoss 不同类别基本损失函数
# parser.add_argument('--base_criterion', type=str, default='BCE3Loss')
parser.add_argument('--base_criterion', type=str, default='BTCB3Loss')
parser.add_argument('--criterion', type=str, default='RRLoss')
# parser.add_argument('--base_criterion', type=str, default='BCETversky3Loss')
# parser.add_argument('--criterion', type=str, default='RRLossNew')
# parser.add_argument('--base_criterion', type=str, default='BCE4LossNew')
# parser.add_argument('--base_criterion', type=str, default='BCETversky4Loss')

# 损失函数 只使用BCE3Loss/BCETversky3Loss，只有单用SMPUNet预测三通道使用这个
# parser.add_argument('--criterion', type=str, default='BCETversky3Loss')
# parser.add_argument('--base_criterion', type=str, default=None)
# parser.add_argument('--criterion', type=str, default='BTCB3Loss')
# parser.add_argument('--criterion', type=str, default='BCETversky2Loss')

# 损失函数 只使用BCETverskyLoss，只有单用SMPUNet或者UNet使用这个,只能用于预测单类别，切换后缀实现
# parser.add_argument('--criterion', type=str, default='BCETverskyLossVT')
# parser.add_argument('--base_criterion', type=str, default=None)

# 单通道循环loss，根据要测的血管选择基础函数，A动脉，V静脉
# parser.add_argument('--criterion', type=str, default='RRLoss')
# parser.add_argument('--base_criterion', type=str, default='BCETverskyLossA')

# 学习率，初始为1e-04
parser.add_argument('--learning_rate', type=float, default=1e-04)

# 递归次数，原始为5
parser.add_argument('--num_iterations', type=int, default=5)
# 网络参数控制
# 输入输出通道，按需调整，使用带有seg的网络要调整到input=6，输出也灵活调整
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
# 每层通道数，原始版本为64，根据显存调整
parser.add_argument('--base_channels', type=int, default=16)

# 编码器名称
parser.add_argument('--encoder_name', type=str, default='resnet50',
                    help='Backbone encoder name for SMP UNet (e.g., resnet18, resnet50, mobilenet_v2)')
# 预训练权重
parser.add_argument('--encoder_weights', type=str, default=None,
                    help='Pretrained weights (e.g., imagenet or None)')
# 编码器层数
parser.add_argument('--encoder_depth', type=int, default=5)

# # 解码器构造 直接使用Basechannel
# parser.add_argument(
#     '--decoder_channels',
#     type=str,
#     default="256, 128, 64, 32, 16",
#     help="Comma-separated list of integers, e.g. '256,128,64,32,16'"
# )

# 训练文件夹数量，包含训练集和验证集为2，改成1启用仅训练模式需要调整train.py
parser.add_argument('--num_folds', type=int, default=2)
# 训练轮次，默认为None
parser.add_argument('--num_epochs', type=int, default=None)
parser.add_argument('--n_proc', type=int, default=1)
# 训练数据挂载文件夹
parser.add_argument('--data_folder', type=str, default='../Data-enhanced')
# 训练命名
parser.add_argument('--version', type=str, default='test' + now_str)
# 训练种子
parser.add_argument('--seed', type=int, default=3)
args = parser.parse_args()

num_folds = args.num_folds
active_folds = range(num_folds)

learning_rate = args.learning_rate
num_epochs = args.num_epochs

dataset = args.dataset

model = args.model
in_channels = args.in_channels
out_channels = args.out_channels
if socket.gethostname() == 'hemingway':
    args.base_channels = 16
base_channels = args.base_channels
num_iterations = args.num_iterations
encoder_name = args.encoder_name
encoder_weights = args.encoder_weights
encoder_depth = args.encoder_depth

criterion = args.criterion
base_criterion = args.base_criterion

n_proc = args.n_proc
gpu_id = args.gpu_id

training_folder = f'../Log/'
version = args.version

seed = args.seed

if dataset == 'GAVE':
    # images = [
    #     'g_001', 'g_002', 'g_003', 'g_004', 'g_005', 'g_006', 'g_007', 'g_008', 'g_009', 'g_010',
    #     'g_011', 'g_012', 'g_013', 'g_014', 'g_015', 'g_016', 'g_017', 'g_018', 'g_019', 'g_020',
    #     'g_021', 'g_022', 'g_023', 'g_024', 'g_025', 'g_026', 'g_027', 'g_028', 'g_029', 'g_030',
    #     'g_031', 'g_032', 'g_033', 'g_034', 'g_035', 'g_036', 'g_037', 'g_038', 'g_039', 'g_040',
    #     'g_041', 'g_042', 'g_043', 'g_044', 'g_045', 'g_046', 'g_047', 'g_048', 'g_049', 'g_050',
    #     'g_051', 'g_052', 'g_053', 'g_054', 'g_055', 'g_056', 'g_057', 'g_058', 'g_059', 'g_060',
    #     'g_061', 'g_062', 'g_063', 'g_064', 'g_065', 'g_066', 'g_067', 'g_068', 'g_069', 'g_070',
    #     'g_071', 'g_072', 'g_073', 'g_074', 'g_075', 'g_076', 'g_077', 'g_078', 'g_079', 'g_080',
    #     'g_081', 'g_082', 'g_083', 'g_084', 'g_085', 'g_086', 'g_087', 'g_088', 'g_089', 'g_090',
    #     'g_091', 'g_092', 'g_093', 'g_094', 'g_095', 'g_096', 'g_097', 'g_098', 'g_099', 'g_100',
    # ]
    images = [
        'g_{:03d}'.format(i) for i in range(1, 51)  # g_001 ~ g_050
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': f'GAVE/training/av',
            'pattern': r'^g_\d{3}\.png$'
        },
        'original': {
            'path': f'GAVE/training/images',
            'pattern': r'^g_\d{3}\.png$'
        },
        'mask': {
            'path': f'GAVE/training/masks',
            'pattern': r'^g_\d{3}\.png$'
        }
    }
elif dataset == 'GAVE+seg':
    images = [
        'g_{:03d}'.format(i) for i in range(1, 51)  # g_001 ~ g_050
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': f'GAVE/training/av',
            'pattern': r'^g_\d{3}\.png$'
        },
        'original': {
            'path': f'GAVE/training/images',
            'pattern': r'^g_\d{3}\.png$'
        },
        'segmentation': {
            'path': f'GAVE/training/segmentation',
            'pattern': r'^g_\d{3}\.png$'
        },
        'mask': {
            'path': f'GAVE/training/masks',
            'pattern': r'^g_\d{3}\.png$'
        }
    }

else:
    raise ValueError('dataset not supported')
