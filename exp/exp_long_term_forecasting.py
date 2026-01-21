from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            #     batch_x = batch_x.float().to(self.device)
            #     batch_y = batch_y.float()
            #     batch_x_mark = batch_x_mark.float().to(self.device)
            #     batch_y_mark = batch_y_mark.float().to(self.device)
            for i, batch_data in enumerate(vali_loader):
                # 1. 动态解包：不管来几个变量，前4个一定是这些
                batch_x = batch_data[0].float().to(self.device)
                batch_y = batch_data[1].float().to(self.device)
                batch_x_mark = batch_data[2].float().to(self.device)
                batch_y_mark = batch_data[3].float().to(self.device)

                # 2. 检查是否有 Anchor (只有 Dataset_Flight 有)
                if len(batch_data) > 4:
                    batch_anchor = batch_data[4].float().to(self.device)
                else:
                    batch_anchor = None  # 其他数据集没有锚点

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            #     iter_count += 1
            #     model_optim.zero_grad()
            #     batch_x = batch_x.float().to(self.device)
            #     batch_y = batch_y.float().to(self.device)
            #     batch_x_mark = batch_x_mark.float().to(self.device)
            #     batch_y_mark = batch_y_mark.float().to(self.device)

            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # 1. 动态解包：不管来几个变量，前4个一定是这些
                batch_x = batch_data[0].float().to(self.device)
                batch_y = batch_data[1].float().to(self.device)
                batch_x_mark = batch_data[2].float().to(self.device)
                batch_y_mark = batch_data[3].float().to(self.device)

                # 2. 检查是否有 Anchor (只有 Dataset_Flight 有)
                if len(batch_data) > 4:
                    batch_anchor = batch_data[4].float().to(self.device)
                else:
                    batch_anchor = None  # 其他数据集没有锚点

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # 定义两个列表，专门用来存归一化的结果
        preds_norm = []
        trues_norm = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                # === 兼容性修改开始 ===
                batch_x = batch_data[0].float().to(self.device)
                batch_y = batch_data[1].float().to(self.device)
                batch_x_mark = batch_data[2].float().to(self.device)
                batch_y_mark = batch_data[3].float().to(self.device)

                # 只有 Dataset_Flight 返回第5个元素
                batch_anchor = batch_data[4].float().to(self.device) if len(batch_data) > 4 else None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                # 保存归一化的输出 (在反归一化和还原之前)
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                preds_norm.append(outputs_np)
                trues_norm.append(batch_y_np)

                # 反归一化 (Inverse Transform)
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    # 此时 outputs 还是 tensor，为了后续运算，这里继续用 numpy 操作比较麻烦，
                    # 建议复用 outputs_np 但需要注意 inverse_transform 需要 2D 输入
                    # 这里保持你原有的逻辑，但在 outputs 变成 numpy 后操作

                    # 注意：上面的 outputs_np 已经转成了 numpy，下面的 outputs 变量在你的原代码里似乎混用了 tensor 和 numpy
                    # 修正逻辑：统一使用 numpy 进行后续处理
                    outputs = outputs_np
                    batch_y = batch_y_np

                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                else:
                    # 如果没有进上面的if，也要确保转成numpy
                    outputs = outputs_np
                    batch_y = batch_y_np

                # === 还原逻辑 (仅当有 anchor 时执行) ===
                if batch_anchor is not None:
                    batch_anchor = batch_anchor.detach().cpu().numpy()
                    batch_anchor = np.expand_dims(batch_anchor, 1)  # [B, 1, C]

                    # 累加还原: Pred_Pos = Anchor + Cumsum(Pred_Delta)
                    outputs = batch_anchor + np.cumsum(outputs, axis=1)
                    batch_y = batch_anchor + np.cumsum(batch_y, axis=1)
                # ======================================

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # ==========================================================
        # 1. 计算并打印归一化指标 (用于论文/学术对比)
        # ==========================================================
        preds_norm = np.concatenate(preds_norm, axis=0)
        trues_norm = np.concatenate(trues_norm, axis=0)

        # 确保形状 [Samples, Pred_Len, Features]
        preds_norm = preds_norm.reshape(-1, preds_norm.shape[-2], preds_norm.shape[-1])
        trues_norm = trues_norm.reshape(-1, trues_norm.shape[-2], trues_norm.shape[-1])

        print("\n" + "=" * 40)
        print(" >>> 归一化指标 (Normalized Metrics) <<<")
        # metric 函数返回: mae, mse, rmse, mape, mspe
        mae_norm, mse_norm, rmse_norm, mape_norm, mspe_norm = metric(preds_norm, trues_norm)
        print(f'Normalized MSE: {mse_norm:.4f}, Normalized MAE: {mae_norm:.4f}')
        print("=" * 40 + "\n")

        # ==========================================================
        # 2. 计算原有物理指标 (用于实际应用分析)
        # ==========================================================
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('Physical Scale - mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('Normalized MSE:{}, Normalized MAE:{} \n'.format(mse_norm, mae_norm))  # 写入归一化结果
        f.write('Physical MSE:{}, Physical MAE:{}, dtw:{}'.format(mse, mae, dtw))  # 写入物理结果
        f.write('\n')
        f.write('\n')
        f.close()

        # 保存 metrics 时，你可以选择保存哪一个，或者都保存
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
