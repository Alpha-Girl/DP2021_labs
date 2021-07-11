import torch
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.Nets import CNNMnist
import random
from phe import *
import numpy as np

class Server():
    def __init__(self, args, w):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        
    def FedAvg(self):
        if self.args.mode == 'plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(
                    update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]

        elif self.args.mode == 'DP':
            C = 2
            simga = 10
            delta = 10 ^ -3
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    # 截断操作
                    update_w_avg[k] += self.clients_update_w[i][k] / \
                        max(1, torch.norm(self.clients_update_w[i][k])/C)
                # 加噪
                update_w_avg[k] = torch.div(
                    update_w_avg[k] + random.gauss(0, simga*C), len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]
        elif self.args.mode == 'Paillier':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    # 加法
                    update_w_avg[k] += self.clients_update_w[i][k]
                # 乘法
                update_w_avg[k] = update_w_avg[k]/len(self.clients_update_w)
            return copy.deepcopy(update_w_avg), sum(self.clients_loss)/len(self.clients_loss)
            '''
            part two: Paillier addition
            '''
        return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

    def test(self, datatest):
        self.model.eval()

        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs,
                                         target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)
                                 ).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss
