# -*- coding: utf-8 -*-
# @Time : 2024/07/17/0017 12:00
# @Author : rainbow
# @Location: 
# @File : train_and_test.py

import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
# from dataset_and_loader import train_data_loader
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import numpy as np
import json
from torchsummary import summary
import os
import datetime
class train(object):
    def __init__(self,model_save_by_loss : bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        now = datetime.datetime.now().strftime('%F_%H-%M-%S')
        self.EPOCHS = 50
        self.batch_size = 2**3
        self.traindatapath = 'Animal_data.txt'
        self.losspath = f'./loss/trainhist-{now}.json'
        self.best_modelpath = f'./model/model-{now}.pth'
        # self.best_modelpath = '/model/best_model-2024-07-05_15-22-22.pth'
        self.best_modelpath_loss = f'./model/model_loss-{now}.pth'
        self.predatapath = f'./outdata/predict-{now}.json'
        self.best_accuracy=0
        self.best_loss = 100
        self.model_save_by_loss = model_save_by_loss
        self.LEARNING_RATE = 2e-5
        self.ADAM_WEIGHT_DECAY = 0
        self.ADAM_BETAS = (0.9, 0.999)
        self.num_workers = 5

        data_load_start = timeit.default_timer()
        # self.traindata,self.valdata,self.testdata=train_data_loader(name_txt=self.traindatapath,
        #                                                             batch_size=self.batch_size,
        #                                                             num_workers=self.num_workers,
        #                                                             shuffle=True)
        transforms_ = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
        ])
        traindata = datasets.MNIST('data', train=True, download=False, transform=transforms_)
        testdata = datasets.MNIST('data', train=False, download=False, transform=transforms_)
        self.traindata = DataLoader(dataset=traindata, batch_size=self.batch_size,num_workers=self.num_workers,shuffle=True)
        self.valdata = DataLoader(dataset=testdata, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True)
        data_load_stop = timeit.default_timer()
        print(f'Data Loading Time:{data_load_stop - data_load_start:.2f}')

        # self.model = model.Net().cuda()
        # self.model = model.FNN(in_features=784,num_class=10).to(self.device)
        summary(self.model,input_size=(1,28,28))
        # self.model = model.CNN(in_channel=3).cuda()
        self.LOSSFUN = nn.CrossEntropyLoss()
        # self.LOSSFUN = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), betas=self.ADAM_BETAS, lr=self.LEARNING_RATE, weight_decay=self.ADAM_WEIGHT_DECAY)
        #
        self.trainhist = {'trainloss': [], 'trainacc': [], 'valloss': [], 'valacc': []}
    def train(self):
        start = timeit.default_timer()
        for epoch in range(1,self.EPOCHS+1):
            train_start = timeit.default_timer()
            self.model.train()
            train_running_loss = 0
            train_labels = []
            train_preds = []
            loop = tqdm(self.traindata,desc=f'EPOCH {epoch}/{int(self.EPOCHS)}', position=0, leave=True)
            for idx, (img,label) in enumerate(loop):
                img = img.to(self.device)
                label = label.to(self.device)
                y_predict = self.model(img).to(self.device)
                y_pred_label_index = torch.argmax(y_predict, dim=1)
                # y_label_index  = torch.argmax(label, dim=1)
                loss = self.LOSSFUN(y_predict,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_running_loss += loss.item()
                train_preds.extend(y_pred_label_index.cpu().detach())
                train_labels.extend(label.cpu().detach())
                loop.set_postfix({'trainloss=':f'{train_running_loss / (idx+1):.3f}',
                               'trainacc=':sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels)})
            train_loss = train_running_loss / (idx + 1)
            train_acc = sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels)
            self.trainhist['trainloss'].append(train_loss)
            self.trainhist['trainacc'].append(train_acc)

            self.model.eval()

            val_running_loss = 0
            val_labels = []
            val_preds = []
            with torch.no_grad():
                loop_ = tqdm(self.valdata,desc=f'EPOCH {epoch}/{int(self.EPOCHS)}', position=0, leave=True)
                for idx, (img, label) in enumerate(loop_):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    y_predict = self.model(img).to(self.device)
                    y_pred_label_index = torch.argmax(y_predict, dim=1)
                    # y_label_index = torch.argmax(label, dim=1)
                    loss = self.LOSSFUN(y_predict, label)
                    val_running_loss += loss.item()
                    val_labels.extend(label.cpu().detach())
                    val_preds.extend(y_pred_label_index.cpu().detach())
                    loop_.set_postfix({'valloss=' : f'{val_running_loss / (idx+1):.3f}',
                                    'valacc=' : sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels)})
                val_loss = val_running_loss / (idx + 1)
                val_acc = sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels)
                self.trainhist['valloss'].append(val_loss)
                self.trainhist['valacc'].append(val_acc)
            train_end = timeit.default_timer()
            print(F'EPOCH {epoch}',f"Training Time:{train_end - train_start:.2f}s",
                  f'trainloss {train_loss:.3f}',f'trainacc {train_acc:.3f}',f'valloss {val_loss:.3f}',f'valacc {val_acc:.3f}')
            if val_acc >self.best_accuracy:
                self.best_accuracy = val_acc
                model_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'trainloss': train_loss,
                    'valloss':val_loss,
                    'trainacc':train_acc,
                    'valacc':val_acc
                }
                torch.save(model_dict,self.best_modelpath)
                print(f'Best model file is saved {self.best_modelpath}')
            while self.model_save_by_loss:
                if val_loss<self.best_loss:
                    self.best_loss = val_loss
                    model_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'trainloss': train_loss,
                        'valloss':val_loss,
                        'trainacc':train_acc,
                        'valacc':val_acc
                    }
                    torch.save(model_dict,self.best_modelpath_loss)
                    print(f'Best model_loss file is saved {self.best_modelpath_loss}')
        # np.save(self.losspath,self.trainhist,allow_pickle=True)
        json.dump(self.trainhist,open(self.losspath,'w'))
        stop = timeit.default_timer()
        print(f"Training Time Sum:{stop - start:.2f}s")
    def test(self):
        test_start = timeit.default_timer()
        data_dicts = {
            'img' : [],
            'y_predict' : [],
            'y_label_index' : [],
            'y_pred_index' :[],
            'accuracy':0
        }
        checkpoint = torch.load(self.best_modelpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        for idx, (img,label) in enumerate(tqdm(self.testdata, position=0, leave=True)):
            img = img.to(self.device)
            label = label.to(self.device)
            y_predict = self.model(img).to(self.device)
            y_pred_label_index = torch.argmax(y_predict, dim=1)
            # y_label_index = torch.argmax(label, dim=1)
            data_dicts['img'].extend(img)
            data_dicts['y_predict'].extend(y_predict)
            data_dicts['y_label_index'].extend(label)
            data_dicts['y_pred_index'].extend(y_pred_label_index)
        data_dicts['accuracy']=sum(1 for x, y in zip(data_dicts['y_pred_index'],
                        data_dicts['y_label_index']) if x == y) / len(data_dicts['y_pred_index'])
        json.dump(data_dicts, open(self.predatapath, 'w'))
        test_stop = timeit.default_timer()
        print(f"Tseting Time Sum:{test_stop - test_start:.2f}s")
if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    trains = train()
    trains.train()
    # trains.test()