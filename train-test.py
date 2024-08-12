# -*- coding: utf-8 -*-
# @Time : 2024/07/14/0014 15:39
# @Author : rainbow
# @Location: 
# @File : train-test.py


import torch
import torch.nn as nn
from torch import optim
import datetime
from tqdm import tqdm
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
# from model import Net,CNN
import numpy as np
from torch.utils.tensorboard import SummaryWriter  
from torchsummary import summary
import json
import os
class Accuracy_Iterator(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.number_correct  = 0
        self.number_total = 0
        self.accuracy  = 0
        self.tmp = None
        
    def update(self,pred_target,target,is_Max : bool = True):
        if is_Max:
            self.tmp = torch.argmax(pred_target,dim=1) 
        else:
            self.tmp = pred_target
        self.number_total +=target.size(0)
        self.number_correct += self.tmp.eq(target.view_as(self.tmp)).sum().item()
        self.accuracy = round(self.number_correct/self.number_total,4)




def training(model,epochs : int,train_data_loader,val_data_loader,
            loss_function,optimizer,device,best_modelpath :str,loss_path :str,
            writer_path :str = None,summary_size  = None,best_accuracy  = 0) -> dict:
    start = datetime.datetime.now().timestamp()
    trainhist = {'trainloss': [], 'trainacc': [], 'valloss': [], 'valacc': []}
    if writer_path is not None:
        writer  = SummaryWriter(writer_path)
    if summary_size is not None:
        summary(model,input_size=summary_size)
    model.train()
    for epoch in range(1,epochs+1):
        train_start = datetime.datetime.now().timestamp()
        train_accuracy = Accuracy_Iterator()

        train_running_loss = 0
        
        loop =tqdm(train_data_loader,desc=f'Epoch {epoch}/{epochs}',position=0,leave=True)
        for idx, (img,target) in enumerate(loop):
            img = img.float().to(device)
            label = target.type(torch.uint8).to(device)
            y_pred = model(img).to(device)
           
            train_accuracy.update(pred_target=y_pred.cpu().detach(),target=label.cpu().detach())
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            
            loop.set_postfix({
                'trainloss':f'{train_running_loss / (idx+1):.3f}',
                'trainacc':train_accuracy.accuracy
            })
        train_loss = train_running_loss / (idx + 1)
        train_acc = train_accuracy.accuracy

        model.eval()
        train_accuracy.reset()
        
        val_running_loss = 0
        with torch.no_grad():
            loop_ = tqdm(val_data_loader,desc=f'Epoch {epoch}/{epochs}', position=0, leave=True)
            for idx, (img ,target)in enumerate(loop_):
                img = img.float().to(device)
                label = target.type(torch.uint8).to(device)
                y_pred = model(img).to(device)
            
                train_accuracy.update(pred_target=y_pred.cpu().detach(),target=label.cpu().detach())
                loss = loss_function(y_pred, label)
                val_running_loss += loss.item()
              
                loop_.set_postfix({
                    'valloss': f'{val_running_loss / (idx + 1):.3f}',
                    'valacc': train_accuracy.accuracy
                })
        val_loss = val_running_loss / (idx + 1)
        val_acc = train_accuracy.accuracy

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': train_loss,
                'valloss': val_loss,
                'trainacc': train_acc,
                'valacc': val_acc
            }
            torch.save(model_dict, best_modelpath)
            print(f'Best model file is saved {best_modelpath}')
        if writer_path is not None:
            writer.add_scalar('trainloss',train_loss,epoch)
            writer.add_scalar('trainaccuracy',train_acc,epoch)
            writer.add_scalar('valloss',val_loss,epoch)
            writer.add_scalar('valaccuracy',val_acc,epoch)
            writer.flush()
        trainhist['trainloss'].append(train_loss)
        trainhist['trainacc'].append(train_acc)
        trainhist['valloss'].append(val_loss)
        trainhist['valacc'].append(val_acc)
        json.dump(trainhist,open(loss_path,'w'))
        train_end = datetime.datetime.now().timestamp()
        print(F'EPOCH {epoch}', f"Training Time:{train_end - train_start:.2f}s",
              f'trainloss {train_loss:.3f}', f'trainacc {train_acc:.4f}', f'valloss {val_loss:.3f}',
              f'valacc {val_acc:.4f}')
    writer.close()
    stop = datetime.datetime.now().timestamp()
    print(f'Train End! Train Sum Time is {stop-start : .2f}s')
    return trainhist
def testing(model,test_data_loader,outdata_path,device) -> dict:
    test_start = datetime.datetime.now().timestamp()
    test_accuracy = Accuracy_Iterator()
    data_dicts = {
        'img': [],
        'y_predict': [],
        'label': [],
        'y_pred_label': [],
        'accuracy': 0
    }
    loop = tqdm(test_data_loader,desc='Testing',position=0,leave=True)
    for img,label in loop:
        img = img.float().to(device)
        label = label.type(torch.uint8).to(device)
        y_pred = model(img).to(device)
        y_pred_label = torch.argmax(y_pred,dim=1)
        test_accuracy.update(pred_target=y_pred_label.cpu().detach(),target=label.cpu().detach(),is_Max=False)
        data_dicts['img'].extend(img.cpu().detach())
        data_dicts['y_predict'].extend(y_pred.cpu().detach())
        data_dicts['label'].extend(label.cpu().detach()) 
        data_dicts['y_pred_label'].extend(y_pred_label.cpu().detach())
    data_dicts['accuracy'] = test_accuracy.accuracy
    json.dump(data_dicts, open(outdata_path, 'w'))
    test_stop = datetime.datetime.now().timestamp()
    print(f"Test End! Test Time Sum:{test_stop - test_start:.2f}s")
    return data_dicts
def train():
    transforms_ = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    now = datetime.datetime.now().strftime('%F_%H-%M-%S')
    epochs = 20
    batch_size = 8
    num_workers = 5
    # model = FNN(in_features=784,num_class=10).to(device)
    # model = Net().to(device)
    # model = CNN(in_channel=1,num_class=10).to(device)
    summary_size = (1,224,224)
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 2e-5
    Adam_Weight_Decay = 0
    Adam_Betas = (0.9, 0.999)
    best_modelpath = f'model/model_{now}_lr{learning_rate}bs{batch_size}.pth'
    loss_path = f'loss/hist_{now}_lr{learning_rate}bs{batch_size}.json'
    writer_path = 'writer/history'
    # optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=Adam_Betas,weight_decay=Adam_Weight_Decay)
    start = datetime.datetime.now().timestamp()

    traindata = datasets.MNIST('data', train=True, transform=transforms_)
    testdata = datasets.MNIST('data', train=False, transform=transforms_)
    trainloader = DataLoader(dataset=traindata, batch_size=batch_size,num_workers = num_workers,pin_memory=False,shuffle=True)
    testloader = DataLoader(dataset=testdata, batch_size=batch_size,num_workers = num_workers,pin_memory=False,shuffle=True)
    
    stop = datetime.datetime.now().timestamp()
    print(f'Data Loading Time:{stop - start:.2f}')
    # training(model=model,epochs=epochs,train_data_loader=trainloader,val_data_loader=testloader,
    #             summary_size=summary_size,loss_function=loss_function,optimizer=optimizer,device=device,
    #             best_modelpath=best_modelpath,loss_path=loss_path,writer_path=writer_path)
    # try:
        
        
    # except:
    #     print('Wraning!! The process ends abnormally!')


if __name__=='__main__':
    '''
    cd /root/autodl-tmp/HWNR_Template
    nohup python -u train-test.py > train.log 2>&1 &
    tail -f train.log
    '''
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    train()
    # transforms_ = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])
    # ])
    # traindata = datasets.MNIST('data', train=True, download=True, transform=transforms_)
    # print()


