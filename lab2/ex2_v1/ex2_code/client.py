
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Nets import CNNMnist
import copy
from phe import *
import numpy as np


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client():

    def __init__(self, args, dataset=None, idxs=None, w=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)

    def train(self):
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)

        net.train()

        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        w_new = net.state_dict()

        delta_w = {}

        if self.args.mode == 'plain':
            for k in w_new.keys():
                delta_w[k] = w_new[k] - w_old[k]

        elif self.args.mode == 'DP':  # DP的client部分与plain相同
            for k in w_new.keys():
                delta_w[k] = w_new[k] - w_old[k]

        elif self.args.mode == 'Paillier':
            for k in w_new.keys():

                delta_w[k] = w_new[k]-w_old[k]
                delta_w[k] = delta_w[k].numpy().astype(float)

        return delta_w, sum(batch_loss) / len(batch_loss)

    def update(self, w_glob):
        self.model.load_state_dict(w_glob)
