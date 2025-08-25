import multiprocessing
import os

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from Hub.tool.train_tool import EarlyStopReduceLROnPlateau, save_model, save_to_csv
from Hub.tool.factory import ModelFactory, LossesFactory


def learning_curves(training, validation, outfile):
    plt.rcParams["figure.figsize"] = [16, 9]
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    assert isinstance(ax1, Axes)
    x, y1 = zip(*training)
    ax1.plot(x, y1, 'b', label='training')

    x, y1 = zip(*validation)
    ax1.plot(x, y1, 'r', label='validation')

    ax1.set_yscale('log')

    ax1.legend()

    fig.savefig(outfile)
    plt.close(fig)


class R2Vessels:

    def __init__(
            self,
            base_channels=64,
            in_channels=3,
            out_channels=3,
            num_iterations=5,
            model=None,
            gpu_id=None,
            criterion=None,
            base_criterion=None,
            learning_rate=1e-4,
            encoder_name='resnet50',
            encoder_weights=None,
            freeze_second_u=False,
            encoder_depth=5
    ):
        self.use_cuda = torch.cuda.is_available()
        self.freeze_second_u = freeze_second_u

        if gpu_id is None:
            self.device = torch.device('cuda', 0)
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cuda', gpu_id)
            torch.cuda.set_device(gpu_id)

        self.criterion_name = criterion
        losses_factory = LossesFactory()
        if base_criterion is not None:
            base_criterion = losses_factory.create_class(base_criterion)
            self.criterion = losses_factory.create_class(criterion, base_criterion=base_criterion)
        else:
            self.criterion = losses_factory.create_class(criterion)

        self.model_name = model
        self.model = ModelFactory().create_class(
            model,
            input_ch=in_channels,
            output_ch=out_channels,
            base_ch=base_channels,
            num_iterations=num_iterations,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            # decoder_channels=decoder_channels
        )

        if freeze_second_u and hasattr(self.model, 'second_u'):
            print("初始化：冻结 second_u 参数")
            for param in self.model.second_u.parameters():
                param.requires_grad = False
        else:
            print("未冻结 second_u 参数")

        if self.use_cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(
            # self.model.parameters(),  # 不冻结参数
            filter(lambda p: p.requires_grad, self.model.parameters()),  # 冻结参数
            lr=learning_rate,
            betas=(0.9, 0.999)
        )
        self.iter = 0

    def dice_coeff(self, predictions, vessels, mask):
        mask = mask[:, 0, :, :]
        mask = torch.round(mask)

        # 确保 predictions 是一个 Tensor,predictions 保存了7个结果,
        # 兼容性部分，确保输入Dice计算的格式对齐
        if isinstance(predictions, list):
            # 取最后一个元素
            pred_vessels = predictions[-1]
        else:
            pred_vessels = predictions  # 已经是 tensor

        # 原始输出结果未经过sigmoid
        pred_vessels = torch.sigmoid(pred_vessels)
        # print("Sigmoid后最大最小值", pred_vessels.max().item(), pred_vessels.min().item())
        pred_vessels = (pred_vessels > 0.5).float()

        # print("Round后非零数量", torch.count_nonzero(pred_vessels))

        def compute_dice(pred, gt, mask):
            pred = pred * mask
            gt = gt * mask
            intersection = (pred * gt).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            return dice.mean().item()

        # 兼容性模块，如果预测结果是三通道
        gt_a = vessels[:, 0, :, :] + vessels[:, 1, :, :]
        gt_vt = vessels[:, 0, :, :] + vessels[:, 1, :, :] + vessels[:, 2, :, :]
        gt_v = vessels[:, 1, :, :] + vessels[:, 2, :, :]

        gt_a = torch.clamp(gt_a, 0, 1)
        gt_vt = torch.clamp(gt_vt, 0, 1)
        gt_v = torch.clamp(gt_v, 0, 1)

        if pred_vessels.size(1) == 3:
            # 计算dice
            pred_a = pred_vessels[:, 0, :, :]
            pred_vt = pred_vessels[:, 1, :, :]
            pred_v = pred_vessels[:, 2, :, :]

            dice_a = compute_dice(pred_a, gt_a, mask)
            dice_vt = compute_dice(pred_vt, gt_vt, mask)
            dice_v = compute_dice(pred_v, gt_v, mask)
        elif pred_vessels.size(1) == 1:
            pred_a = pred_vessels[:, 0, :, :]
            pred_vt = pred_vessels[:, 0, :, :]
            pred_v = pred_vessels[:, 0, :, :]
            dice_a = compute_dice(pred_a, gt_a, mask)
            dice_vt = compute_dice(pred_vt, gt_vt, mask)
            dice_v = compute_dice(pred_v, gt_v, mask)
        else:
            pred_a = pred_vessels[:, 0, :, :]
            pred_v = pred_vessels[:, 1, :, :]
            dice_a = compute_dice(pred_a, gt_a, mask)
            dice_v = compute_dice(pred_v, gt_v, mask)
            dice_vt = 0.0

        return [dice_a, dice_vt, dice_v]

    def train_epoch(self, r2v_loader):
        total_loss = 0.0
        total_dice = []  # A, VT, V

        self.model.train()

        for sample in r2v_loader:
            data = sample[1]

            retino = data[0].cuda(non_blocking=True).requires_grad_(True)
            vessels = data[1].cuda(non_blocking=True).requires_grad_(False)
            mask = data[2].cuda(non_blocking=True).requires_grad_(False)

            self.optimizer.zero_grad()

            predictions = self.model(retino)  #
            loss = self.criterion(predictions, vessels, mask)
            # 计算Dice
            dice = self.dice_coeff(predictions, vessels, mask)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            total_dice.append(dice)

            self.iter += 1

        pattern = '\n|{}| [{}, {}] >> training epoch mean loss: {}, Dice:VT {:.4f}//A {:.4f}//V {:.4f}'
        avg_loss = total_loss / len(r2v_loader)
        total_dice_tensor = torch.tensor(total_dice)  # shape: [N, 3]
        avg_dice = total_dice_tensor.mean(dim=0).tolist()  # list: [avg_a, avg_vt, avg_v]
        print(pattern.format(
            self.iter,
            self.model_name,
            self.criterion_name,
            avg_loss,
            avg_dice[1],
            avg_dice[0],
            avg_dice[2]
        ))

        return [avg_loss], avg_dice

    def test(self, r2v_dataloader, prefix_to_save=None):
        with torch.no_grad():
            total_loss = 0.0
            total_dice = []  # A, VT, V

            self.model.eval()

            for sample in r2v_dataloader:
                try:
                    k = sample[0].numpy()[0]
                except AttributeError:
                    k = sample[0][0]
                data = sample[1]

                retino = data[0].cuda(non_blocking=True)
                vessels = data[1].cuda(non_blocking=True)
                mask = data[2].cuda(non_blocking=True)

                predictions = self.model(retino)
                loss = self.criterion(predictions, vessels, mask)

                # 计算dice
                dice = self.dice_coeff(predictions, vessels, mask)

                if prefix_to_save is not None:
                    self.criterion.save_predicted(predictions, prefix_to_save + str(k) + '.png')

                total_loss += loss.item()
                total_dice.append(dice)

            pattern = '\n|{}| [{}, {}] >> validation epoch mean loss: {}, Dice:VT {:.4f}//A {:.4f}//V {:.4f}'
            avg_loss = total_loss / len(r2v_dataloader)
            total_dice_tensor = torch.tensor(total_dice)  # shape: [N, 3]
            avg_dice = total_dice_tensor.mean(dim=0).tolist()  # list: [avg_a, avg_vt, avg_v]
            print(pattern.format(
                self.iter,
                self.model_name,
                self.criterion_name,
                avg_loss,
                avg_dice[1],
                avg_dice[0],
                avg_dice[2]
            ))

            return [avg_loss], avg_dice

    def training(
            self,
            train_loader,
            test_loader,
            path_to_save,
            init_iter=1,
            save_period=25,
            scheduler_patience=25,
            stopping_patience=25
    ):
        save_to_csv([['best_loss', 'iter', 'best_dice']], os.path.join(path_to_save, 'best_loss.csv'))
        save_to_csv([['loss', 'iter', 'dice']], os.path.join(path_to_save, 'train_loss.csv'))
        save_to_csv([['loss', 'iter', 'dice']], os.path.join(path_to_save, 'test_loss.csv'))
        train_loss = list()
        test_loss = list()
        train_dice = list()
        test_dice = list()
        all_train_loss = list()
        all_test_loss = list()
        all_train_dice = list()
        all_test_dice = list()

        scheduler = EarlyStopReduceLROnPlateau(
            self.optimizer,
            self.model,
            path_to_save,
            factor=0.1,
            patience=scheduler_patience,
            patience_stopping=stopping_patience,
            verbose=True,
            cooldown=0,
            threshold=0,
            min_lr=1e-8,
            eps=0
        )

        self.iter = init_iter

        test_count = 1
        while scheduler.training():

            # 解冻控制逻辑
            if self.iter == 8001 and hasattr(self.model, 'second_u') and self.freeze_second_u:
                print(f"[Iter {self.iter}] Warmup阶段结束，解冻 second_u 并重建优化器")
                for param in self.model.second_u.parameters():
                    param.requires_grad = True

                # 重建优化器，加入新的可训练参数
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.optimizer.param_groups[0]['lr'],  # 保持之前的学习率
                    betas=(0.9, 0.999)
                )

            save = (test_count % save_period == 0) or test_count == 1

            if save and self.iter >= 10000:
                save_path = os.path.join(path_to_save, "Pred", str(self.iter))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                prefix_to_save = save_path + '/'
            else:
                prefix_to_save = None
            # Train and test for each epoch, storing both loss and dice
            train_loss_epoch, train_dice_epoch = self.train_epoch(train_loader)
            test_loss_epoch, test_dice_epoch = self.test(test_loader, prefix_to_save)

            train_loss.append([self.iter] + train_loss_epoch)
            test_loss.append([self.iter] + test_loss_epoch)
            train_dice.append([self.iter] + [train_dice_epoch])
            test_dice.append([self.iter] + [test_dice_epoch])

            is_best = scheduler.step(test_loss[-1][1], self.iter)

            if is_best:
                save_to_csv([[str(test_loss[-1][1]), str(self.iter)]],
                            os.path.join(path_to_save, 'best_loss.csv'))
                save_model(self.model, path_to_save + '/generator_best.pth')

            if save:
                save_to_csv(train_loss, os.path.join(path_to_save, 'train_loss.csv'))
                save_to_csv(test_loss, os.path.join(path_to_save, 'test_loss.csv'))
                save_to_csv(train_dice, os.path.join(path_to_save, 'train_dice.csv'))
                save_to_csv(test_dice, os.path.join(path_to_save, 'test_dice.csv'))
                all_train_loss += train_loss
                all_test_loss += test_loss
                all_train_dice += train_dice
                all_test_dice += test_dice
                train_loss = []
                test_loss = []
                train_dice = []
                test_dice = []
                learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')

            test_count += 1

        if len(train_loss) > 0:
            save_to_csv(train_loss, os.path.join(path_to_save, 'train_loss.csv'))
        if len(test_loss) > 0:
            save_to_csv(test_loss, os.path.join(path_to_save, 'test_loss.csv'))
        if len(train_dice) > 0:
            save_to_csv(train_dice, os.path.join(path_to_save, 'train_dice.csv'))
        if len(test_dice) > 0:
            save_to_csv(test_dice, os.path.join(path_to_save, 'test_dice.csv'))
        save_model(self.model, path_to_save + '/generator_last.pth')
        learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')
