import torch
import torch.nn as nn
import os
# from sklearn.datasets import make_regression
# from torch.utils.data import Dataset, DataLoader
import numpy as np


class EarlyStopping: # 这个是别人写的工具类，大家可以把它放到别的地方
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, verbose=False, delta=0.01, model_path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = model_path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience},best `score is {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter = 0
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# class MyDataSet(Dataset):  # 定义数据格式
#     def __init__(self, train_x, train_y, sample):
#         self.train_x = train_x
#         self.train_y = train_y
#         self._len = sample

#     def __getitem__(self, item: int):
#         return self.train_x[item], self.train_y[item]

#     def __len__(self):
#         return self._len


# def get_data():
#     """构造数据"""
#     sample = 20000
#     data_x, data_y = make_regression(n_samples=sample, n_features=100)  # 生成数据集
#     train_data_x = data_x[:int(sample * 0.8)]
#     train_data_y = data_y[:int(sample * 0.8)]
#     valid_data_x = data_x[int(sample * 0.8):]
#     valid_data_y = data_y[int(sample * 0.8):]
#     train_loader = DataLoader(MyDataSet(train_data_x, train_data_y, len(train_data_x)), batch_size=10)
#     valid_loader = DataLoader(MyDataSet(valid_data_x, valid_data_y, len(valid_data_x)), batch_size=10)
#     return train_loader, valid_loader


# class LinearRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)  # 输入的个数，输出的个数

#     def forward(self, x):
#         out = self.linear(x)
#         return out


def main():
    train_loader, valid_loader = get_data()
    model = LinearRegressionModel(input_dim=100, output_dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=4, verbose=True)  # 早停

    # 开始训练模型
    for epoch in range(1000):
        # 正常的训练
        print("迭代第{}次".format(epoch))
        model.train()
        train_loss_list = []
        for train_x, train_y in train_loader:
            optimizer.zero_grad()
            outputs = model(train_x.float())
            loss = criterion(outputs.flatten(), train_y.float())
            loss.backward()
            train_loss_list.append(loss.item())
            optimizer.step()
        print("训练loss:{}".format(np.average(train_loss_list)))
        # 早停策略判断
        model.eval()
        with torch.no_grad():
            valid_loss_list = []
            for valid_x, valid_y in valid_loader:
                outputs = model(valid_x.float())
                loss = criterion(outputs.flatten(), valid_y.float())
                valid_loss_list.append(loss.item())
            avg_valid_loss = np.average(valid_loss_list)
            print("验证集loss:{}".format(avg_valid_loss))
            early_stopping(avg_valid_loss, model)
            if early_stopping.early_stop:
                print("此时早停！")
                break


# if __name__ == '__main__':
#     main()
