
import random
import numpy as np
from sklearn.metrics import f1_score, recall_score
import torch
import torch.nn as nn
# from torch import optim
import datetime
from tqdm import tqdm
# from torchvision import datasets,transforms
# from torch.utils.data import Dataset,DataLoader
from torchvision.utils import  make_grid
import numpy as np
from torch.utils.tensorboard import SummaryWriter  
from torchsummary import summary
import json
import pickle
import os
import pandas
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
import math
########################################################### Seed set##########################################################
def set_same_seed(seed):
    """
        定义随机种子固定
    """
    print('seed:',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    """
        定义Dataloader随机种子
    """
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def seed_worker(worker_id):
    """
        定义Dataloader随机种子  官方版
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    print(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

######################################################parameters update ####################################################################
class MetricMonitor(object):
    """
    计算训练过程中准确率的变化。适用于分类任务
    reset:
        重置计算器
    update：
        更新准确率,召回率,F1分数的平均值
    """
    def __init__(self):
        self.reset()
    def reset(self):  
        # self.accuracy_tmp  = {'number_correct'  : 0,'number_total'  :0}
        self.number_correct  = 0
        self.number_total = 0
        self.batch_num = 0
        self.recallsum = 0
        self.f1sum = 0
        self._Accuracy_ = 0
        self._Recallscore_ = 0
        self._F1score_ = 0
        self.tmp = None
        
    def update(self,pred_target,target,is_Max : bool = True,updata_recall:bool = False,update_f1:bool = False):
        self.tmp = torch.argmax(pred_target,dim=1) if is_Max else pred_target
        self.number_total +=target.size(0) #更新训练数据数量
        batch_correct = self.tmp.eq(target.view_as(self.tmp)).sum().item()#统计预测结果与真实结果相同的个数
        self.number_correct += batch_correct
        self._Accuracy_ = self.number_correct/self.number_total #计算准确率
        if updata_recall or update_f1 :
            self.batch_num+=1
            recall_score_ =recall_score(y_true=target,y_pred=self.tmp,average="macro", zero_division=0)##计算召回率
            self.recallsum+=recall_score_
            self._Recallscore_ = self.recallsum/self.batch_num
            if update_f1:
                self.f1sum+=f1_score(y_true=target,y_pred=self.tmp,average='macro') ##计算F1分数
                self._F1score_ = self.f1sum/self.batch_num
    def __str__(self) -> str:
        return " | ".join([
                f'Accuracy:{self._Accuracy_}',f'Recall:{self._Recallscore_}',f'F1:{self._F1score_}'
        ])
class DataUpdate(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.metric = {}
    def update(self,**kwargs):
        for key,value in kwargs.items():
            if key not in self.metric.keys():
                self.metric[str(key)] = []
            if isinstance(value,list):
                self.metric[str(key)].extend(value)
            else:
                self.metric[str(key)].append(value)
########################################################## lr adjust################################################################
""" learning rate schedule """
def Adjust_learning_rate_consine(optimizer, epoch, init_lr,min_lr,n_epochs, ibatch, NBatch,num_init : int  =1):
    """ adjust learning of a given optimizer and return the new learning rate """
    """
        optimizer : 优化器
        epoch:当前轮次
        init_lr: 初始学习率
        n_epochs:总轮次
        ibatch : 数据加载器Dataloader的批次
        NBatach : 训练数据数量 
        $$公式
        lr_{new}\,\,=\,\,\frac{1}{2}lr_{out}\left( 1+\cos \left( \frac{\pi *\left( epoch*N+ibatch \right)}{nepochs*N} \right) \right) 
        $$
    """
    t_total = n_epochs * NBatch
    t_cur = epoch * NBatch + ibatch
    # new_lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    new_lr =min_lr+ 0.5 * (init_lr-min_lr) * (1 + math.cos(num_init*math.pi * t_cur / t_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr
def Adjust_learning_rate_exponent(optimizer,epoch,max_lr,min_lr,Tau,nEpoch, iBatch, nBatch,mode):
    """
        指数函数学习率衰减
        optimizer : 优化器
        epoch:当前轮次从1开始
        max_lr: 初始学习率
        min_lr : 最小学习率
        basenum:指数项分母
        tau : 衰减指数
        ibatch : 数据加载器Dataloader的批次
        NBatach : 训练数据数量  
    """
    assert epoch>=1,'Parameter\'epoch\' should be more than 1!'
    # assert iBatch>=1,'Parameter\'iBatch\' should be more than 1!'
    t_total = nEpoch * nBatch
    t_cur = (epoch-1) * nBatch + iBatch
    exponent = t_cur/t_total
    if mode=='down':
        new_lr = min_lr+(max_lr-min_lr)*(Tau-Tau**exponent)/(Tau-1)
    elif mode=='up':
        new_lr = min_lr+(max_lr-min_lr)*(Tau**exponent-1)/(Tau-1)
    else:
        raise ValueError("The mode does not exist!")
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr




######################################################### PLOT #################################################################
def imgshow(img,pathsave=None,title = None,denorm :list[list] = None,is_show : bool = True,is_save : bool= False):
    plt.figure(figsize=(16,8))
    img = img.numpy().transpose(1,2,0)
    if denorm is not None:
        mean = np.array(denorm[0])
        std = np.array(denorm[1])
        img = mean+std*img
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    if is_save and pathsave is not None :
        # 
        plt.savefig(pathsave,dpi = 600, bbox_inches='tight')
    if is_show:plt.show() 
    else: 
        plt.close()
def batch_imgshow(data,target,predtarget,batch_size,denrom = None,filename=None,pathsave = None,is_show : bool = True,is_save : bool= False):
    data = make_grid(data,nrow=int(batch_size//2),padding=2)
    imgshow(img = data,title=str(filename)+'\nTrue:'+str(target)+'\nPred:'+str(predtarget),
            pathsave=pathsave,is_save=is_save,is_show=is_show,denorm=denrom)
    

def pltsetFont(fontsize =12):
    plt.rcParams['font.family'] = ['Times New Roman','SimSun']#设置全局字体
    plt.rcParams['axes.titlesize'] = fontsize       # 设置标题字体大小
    plt.rcParams['axes.titleweight'] = 'bold'  # 设置标题字体为加粗
    plt.rcParams['axes.labelsize'] = fontsize   #轴标题字体大小
    plt.rcParams['axes.labelweight'] = 'bold'  #轴标题字体粗细
    plt.rcParams['xtick.labelsize'] = fontsize  #x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = fontsize  #y轴刻度字体大小
    plt.rcParams['axes.linewidth'] = 1         # 设置轴的线宽，也会影响刻度线
    plt.rcParams['xtick.direction'] = 'out'     # 设置x轴刻度方向为向外
    plt.rcParams['ytick.direction'] = 'out'     # 设置y轴刻度方向为向外
    plt.rcParams['legend.fontsize']= fontsize



def Trainresult_Plottoshow(datadict,path=None,toshow: bool = False):
    # assert path.split('.')[-1]=='json' ,'Inputing file is not in ".json" format!'
    # print('Loading data')
    # with open(path,'r') as data:
    #     datadict = json.load(data)
    # print('Loading data done')
    '''
        'trainloss': [],
        'trainacc': [], 
        'valloss': [], 
        'valacc': [],
        'trainrecallscore':[],
        'trainf1score':[],
        'valrecallscore':[],
        'valf1score':[]
    '''
    trainloss =datadict['trainloss']
    trainacc = datadict['trainacc']
    trainrecall = datadict['trainrecallscore']
    trainf1 = datadict['trainf1score']
    valloss = datadict['valloss']
    valacc = datadict['valacc']
    valrecall = datadict['valrecallscore']
    valf1 = datadict['valf1score']
    plt.figure(figsize=(12,6))
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    plt.subplot(2,2,1)
    plt.plot(trainloss,color = 'r',label = 'trainloss')
    plt.plot(valloss,color = 'g',label = 'valloss')
    plt.title(f'Loss\nmin loss: trainloss {np.min(trainloss):.3f} valloss {np.min(valloss):.3f}',fontdict={'fontsize':12})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(trainacc,color = 'r',label = 'trainacc')
    plt.plot(valacc,color = 'g',label = 'valacc')
    plt.title(f'Accuracy\nmax acc: trainacc {np.max(trainacc):.3f} valacc {np.max(valacc):.3f}' ,fontdict={'fontsize':12})
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(trainrecall,color = 'r',label = 'trainrecallscore')
    plt.plot(valrecall,color = 'g',label = 'valrecallscore')
    plt.title(f'Recall\nmax recall:trainrecall {np.max(trainrecall):.3f} valrecall {np.max(valrecall):.3f}',fontdict={'fontsize':12})
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(trainf1,color = 'r',label = 'trainf1score')
    plt.plot(valf1,color = 'g',label = 'valf1score')
    plt.title(f'F1\nmax F1:trainF1 {np.max(trainf1):.3f} valF1 {np.max(valf1):.3f}',fontdict={'fontsize':12})
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    if path is not None:
        plt.savefig(path,dpi = 600, bbox_inches='tight')
    if toshow :plt.show() 
    else :plt.close()
############################################### TRAIN  ##########################################################################

def Train_epoch(model,
                train_data_loader,
                optimizer,
                loss_function,
                MetricMonitors,
                device)->tuple:
    '''
        适用于分类任务
    '''
    model.train()
    loss =0
    with tqdm(train_data_loader,desc='Training',position=0,unit='batch',leave=True) as loop:
        for index,(inputs,targets) in enumerate(loop):
            inputs  = inputs.to(device,non_blocking = True)
            targets = targets.to(device,non_blocking = True)
            predict_targets = model(inputs).to(device,non_blocking = True)
            loss_present = loss_function(predict_targets,targets)
            MetricMonitors.update(predict_targets.cpu().detach(),
                                  targets.cpu().detach(),True,True,True)
            optimizer.zero_grad()
            loss_present.backward()
            optimizer.step()

            loss+=loss_present.item()
            loop.set_postfix(
                {
                   'loss':loss/(index+1),
                   'accuracy':MetricMonitors._Accuracy_,
                   'recallscore':MetricMonitors._Recallscore_,
                   'f1score':MetricMonitors._F1score_
                }
            )
    loss = loss/len(train_data_loader)
    acc = MetricMonitors._Accuracy_
    recall = MetricMonitors._Recallscore_
    f1 = MetricMonitors._F1score_
    return loss,acc,recall,f1
def Evaluate_epoch(model,
             val_data_loader,
             loss_function,
             MetricMonitors,
             device)->tuple:
    """
        适用于分类任务
    """
    model.eval()
    MetricMonitors.reset()
    loss = 0
    with torch.no_grad():
        with tqdm(val_data_loader,desc='Evaluating',position=0,unit='batch',leave=True) as loop_:
             for index,(inputs,targets) in enumerate(loop_):
                inputs  = inputs.to(device,non_blocking = True)
                targets = targets.to(device,non_blocking = True)
                predict_targets = model(inputs).to(device,non_blocking = True)
                loss_present = loss_function(predict_targets,targets)
                MetricMonitors.update(predict_targets.cpu().detach(),
                                        targets.cpu().detach(),True,True,True)
                loss+=loss_present.item()
                loop_.set_postfix({
                    'loss':loss/(index+1),
                    'accuracy':MetricMonitors._Accuracy_,
                    'recallscore':MetricMonitors._Recallscore_,
                    'f1score':MetricMonitors._F1score_
                })
    loss = loss/len(val_data_loader)
    acc = MetricMonitors._Accuracy_
    recall = MetricMonitors._Recallscore_
    f1 = MetricMonitors._F1score_
    return loss,acc,recall,f1
def model_save(bestmodel_path,
               lastmodel_path,
               model_paramters,
               val_acc,best_acc):
    '''
    model_paramters = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': train_loss,
                'valloss': val_loss,
                'trainacc': train_acc,
                'valacc': val_acc,
                'trainrecallscore':trainrecallscore,
                'trainf1score':trainf1score,
                'valrecallscore':valrecallscore,
                'valf1score':valf1score,
            }
    '''
    
    torch.save(model_paramters,lastmodel_path)
    print(f'The last Model has been saved \'{lastmodel_path}\'!!')
    if best_acc<val_acc:
        best_acc = val_acc
        torch.save(model_paramters,bestmodel_path)
        print(f'The best Model has been saved \'{bestmodel_path}\' as acc={best_acc}!!')
    return best_acc

def To_save_csv(datadict,path):
    maxlength = np.max([len(value) for value in datadict.values()])
    for value in datadict.values():
        value.extend(['']*(maxlength-len(value)))
    # print(datadict)
    data = pandas.DataFrame(datadict)
    data.to_csv(path, index=True)

def Train(model,
          epochs,
          train_data_loader,
          val_data_loader,
          optimizer,
          loss_function,
          modelframeworker_path,
          result_path,
          lastmodel_path,
          bestmodel_path,
          resultfig_path,
          device,
          lradjust : list = None,):
    start = datetime.datetime.now().timestamp()
    # json.dump(str(model),open(modelframeworker_path,'w'))
    with open(modelframeworker_path, 'w', encoding='utf-8') as file:
        file.write(str(model))
    # global best_acc
    best_acc=0
    trainmetric = DataUpdate()
    for epoch in range(1,epochs+1):
        train_start = datetime.datetime.now().timestamp()
        MetricMonitors = MetricMonitor()
        print(f'Epoch {epoch}/{epochs}:')
        train_loss,train_acc,train_recall,train_f1=Train_epoch(
            model=model,
            train_data_loader=train_data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            MetricMonitors=MetricMonitors,
            device=device
        )
        if lradjust is not None:
            new_lr = Adjust_learning_rate_exponent(optimizer,epoch,lradjust[1],
                                                        lradjust[0],lradjust[2],
                                                        epochs,0,1,mode=lradjust[3])
        val_loss,val_acc,val_recall,val_f1 = Evaluate_epoch(
            model=model,
            val_data_loader=val_data_loader,
            loss_function=loss_function,
            MetricMonitors=MetricMonitors,
            device=device
        )
        model_paramters = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': train_loss,
                'valloss': val_loss,
                'trainacc': train_acc,
                'valacc': val_acc,
                'trainrecallscore':train_recall,
                'trainf1score':train_f1,
                'valrecallscore':val_recall,
                'valf1score':val_f1,
            }
        best_acc = model_save(bestmodel_path,lastmodel_path,model_paramters,val_acc,best_acc)

        trainmetric.update(
            trainloss = train_loss,
            trainacc = train_acc,
            trainrecallscore = train_recall,
            trainf1score = train_f1,
            valloss=val_loss,
            valacc=val_acc,
            valrecallscore=val_recall,
            valf1score=val_f1,
        )

        To_save_csv(trainmetric.metric,result_path)
        train_end = datetime.datetime.now().timestamp()
        print(F'EPOCH {epoch}', 
              f"Training Time:{train_end - train_start:.2f}s",
              f'lr {new_lr: .6f}',
              f'trainloss {train_loss:.6f}',
              f'trainacc {train_acc:.6f}',
              f'valloss {val_loss:.6f}',
              f'valacc {val_acc:.6f}')
    # pltsetFont(fontsize=12)
    Trainresult_Plottoshow(trainmetric.metric,resultfig_path)
    stop = datetime.datetime.now().timestamp()
    print(f'Train End! Train Sum Time is {(stop-start)/60: .3f}min')
def Test(model,
         modelpath,
         test_data_loader,
         outdata_path,
         device,):
    MetricMonitors = MetricMonitor()
    start = datetime.datetime.now().timestamp()
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    inputlist = []
    valmetric = DataUpdate()
    model.eval()
    with torch.no_grad():
        with tqdm(test_data_loader,desc='Testing',position=0,unit='batch',leave=True) as loop:
            for inputs,targets in loop:
                inputs  = inputs.to(device,non_blocking = True)
                targets = targets.to(device,non_blocking = True)
                predict_targets = model(inputs).to(device,non_blocking = True)
                MetricMonitors.update(predict_targets.cpu().detach(),targets.cpu().detach())
                loop.set_postfix({'Accuracy':MetricMonitors._Accuracy_})

                inputlist.extend(inputs.cpu().detach().tolist())
                valmetric.update(
                    targets = targets.cpu().detach().tolist(),
                    predict_targets=torch.argmax(predict_targets,dim=1).cpu().detach().tolist()
                )
            valmetric.update(accuracy = MetricMonitors._Accuracy_)
    print('Saving!')
    # json.dump(data_dicts, open(outdata_path, 'w'))
    To_save_csv(valmetric.metric,outdata_path)
    pickle.dump(inputlist,open(str(outdata_path[:-len(outdata_path.split('.')[-1])]+'dat'),'wb'))
    print('Saving Done!')
    stop = datetime.datetime.now().timestamp()
    print(f"Test End! Test Time Sum:{stop - start:.2f}s")
    return valmetric.metric



def training(model,epochs : int,
             train_data_loader,
             val_data_loader,
             loss_function,
             optimizer,
             device,
             best_modelpath :str,
             loss_path :str,
             Adjust_lr:dict = None,
             writer_path :str = None,
             summary_size  = None,
             best_accuracy  = 0) -> dict:
    start = datetime.datetime.now().timestamp()
    trainhist = {'trainloss': [], 'trainacc': [], 'valloss': [], 'valacc': [],
                 'trainrecallscore':[],'trainf1score':[],'valrecallscore':[],'valf1score':[]}
    if writer_path is not None:
        writer  = SummaryWriter(writer_path)
    if summary_size is not None:
        summary(model,input_size=summary_size)
    nBatch = len(train_data_loader)
    model.train()
    for epoch in range(1,epochs+1):
        train_start = datetime.datetime.now().timestamp()
        train_Monitor = MetricMonitor()

        train_running_loss = 0
        # trainlabel = []
        # trainpred  = []
        # trainrecall = []
        loop =tqdm(train_data_loader,desc=f'Epoch {epoch}/{epochs}',position=0,leave=True)
        for idx, (img,target) in enumerate(loop):
            img = img.float().to(device,non_blocking=True)
            label = target.type(torch.uint8).to(device,non_blocking=True)
            y_pred = model(img).to(device,non_blocking=True)
            y_pred_label = torch.argmax(y_pred,dim=1)
            # trainrecall.append(recall_score(label.cpu().detach(),y_pred_label.cpu().detach(),average='macro',zero_division=0))
            # trainlabel.extend(label.cpu().detach())
            # trainpred.extend(y_pred_label.cpu().detach())
            train_Monitor.update(pred_target=y_pred.cpu().detach(),target=label.cpu().detach(),update_f1=True)
            loss = loss_function(y_pred, label)
            if Adjust_lr is not None:
                newlr = Adjust_learning_rate_exponent(optimizer=optimizer,epoch=epoch,max_lr = Adjust_lr['max_lr'],
                                              min_lr=Adjust_lr['min_lr'],Tau = Adjust_lr['tau'],
                                              nEpoch=epochs,iBatch=idx,nBatch=nBatch,mode = Adjust_lr['mode'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            

            loop.set_postfix({
                'trainloss':f'{train_running_loss / (idx+1):.3f}',
                'trainacc':train_Monitor._Accuracy_,
                'lr':newlr,
                'Recallscore':train_Monitor._Recallscore_,
                'F1score':train_Monitor._F1score_
            })
        train_loss = train_running_loss / (idx + 1)
        train_acc = train_Monitor._Accuracy_
        trainrecallscore = train_Monitor._Recallscore_
        trainf1score = train_Monitor._F1score_
        trainhist['trainloss'].append(train_loss)
        trainhist['trainacc'].append(train_acc)
        trainhist['trainrecallscore'].append(train_Monitor._Recallscore_)
        trainhist['trainf1score'].append(train_Monitor._F1score_)
        # recall1 = np.average(trainrecall)
        # f1 = f1_score(trainlabel,trainpred,average='macro')
        model.eval()
        train_Monitor.reset()
        
        val_running_loss = 0
        with torch.no_grad():
            loop_ = tqdm(val_data_loader,desc=f'Epoch {epoch}/{epochs}', position=0, leave=True)
            for idx, (img ,target)in enumerate(loop_):
                img = img.float().to(device,non_blocking=True)
                label = target.type(torch.uint8).to(device,non_blocking=True)
                y_pred = model(img).to(device,non_blocking=True)
            
                train_Monitor.update(pred_target=y_pred.cpu().detach(),target=label.cpu().detach(),update_f1=True)
                loss = loss_function(y_pred, label)
                val_running_loss += loss.item()
              
                loop_.set_postfix({
                    'valloss': f'{val_running_loss / (idx + 1):.3f}',
                    'valacc': train_Monitor._Accuracy_,
                    'Recallscore':train_Monitor._Recallscore_,
                    'F1score':train_Monitor._F1score_
                })
        val_loss = val_running_loss / (idx + 1)
        val_acc = train_Monitor._Accuracy_

        trainhist['valloss'].append(val_loss)
        trainhist['valacc'].append(val_acc)

        trainhist['valrecallscore'].append(train_Monitor._Recallscore_)
        trainhist['valf1score'].append(train_Monitor._F1score_)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': train_loss,
                'valloss': val_loss,
                'trainacc': train_acc,
                'valacc': val_acc,
                'trainrecallscore':trainrecallscore,
                'trainf1score':trainf1score,
                'valrecallscore':train_Monitor._Recallscore_,
                'valf1score':train_Monitor._F1score_,
                'lr':newlr
            }
            torch.save(model_dict, best_modelpath)
            print(f'Best model file is saved {best_modelpath}')
        if writer_path is not None:
            writer.add_scalar('trainloss',train_loss,epoch)
            writer.add_scalar('trainaccuracy',train_acc,epoch)
            writer.add_scalar('valloss',val_loss,epoch)
            writer.add_scalar('valaccuracy',val_acc,epoch)
            writer.add_scalar('trainrecallscore',trainrecallscore,epoch)
            writer.add_scalar('trainf1score',trainf1score,epoch)
            writer.add_scalar('valrecallscore',train_Monitor._Recallscore_,epoch)
            writer.add_scalar('valf1score',train_Monitor._F1score_,epoch)
            writer.flush()
        
        json.dump(trainhist,open(loss_path,'w'))
        train_end = datetime.datetime.now().timestamp()
        print(F'EPOCH {epoch}', f"Training Time:{train_end - train_start:.2f}s",
              f'trainloss {train_loss:.6f}', f'trainacc {train_acc:.4f}', f'valloss {val_loss:.6f}',
              f'valacc {val_acc:.4f}')
    writer.close()
    stop = datetime.datetime.now().timestamp()
    print(f'Train End! Train Sum Time is {stop-start : .2f}s')
    return trainhist
def testing(model,
            test_data_loader,
            outdata_path:str,
            device,
            pathsavei:str=None,
            class_dict:dict=None,
            denorm = None) -> dict:
    if pathsavei is not None:
        if not os.path.exists(pathsavei):os.mkdir(pathsavei)
    test_start = datetime.datetime.now().timestamp()
    test_accuracy = MetricMonitor()
    data_dicts = { 
        'img': [],
        'y_predict': [],
        'label': [],
        'y_pred_label': [],
        'accuracy': 0
    }
    loop = tqdm(test_data_loader,desc='Testing',position=0,leave=True)
    for idx ,(img,target) in enumerate(loop):
        img = img.float().to(device,non_blocking=True)
        label = target.type(torch.uint8).to(device,non_blocking=True)
        y_pred = model(img).to(device,non_blocking=True)
        y_pred_label = torch.argmax(y_pred,dim=1)
        if class_dict is not None and pathsavei is not None:
            pathsave = pathsavei+f'batch{idx}.jpg'
            predtarget = [class_dict[i] for i in y_pred_label.cpu().detach().tolist()]
            batch_imgshow(data=img.cpu(),pathsave=pathsave,batch_size=img.size(0),denrom=denorm,predtarget=predtarget,
                        filename='Result',target=target.cpu().detach().tolist(),is_save=True,is_show=False)
        test_accuracy.update(pred_target=y_pred_label.cpu().detach(),target=label.cpu().detach(),is_Max=False)
        loop.set_postfix({'Accuracy':test_accuracy._Accuracy_})
        # data_dicts['img'].extend(target)
        data_dicts['y_predict'].extend(y_pred.cpu().detach().tolist())
        data_dicts['label'].extend(label.cpu().detach().tolist()) 
        data_dicts['y_pred_label'].extend(y_pred_label.cpu().detach().tolist())
    data_dicts['accuracy'] = test_accuracy._Accuracy_
    print('Saving!')
    json.dump(data_dicts, open(outdata_path, 'w'))
    print('Saving Done!')
    test_stop = datetime.datetime.now().timestamp()
    print(f"Test End! Test Time Sum:{test_stop - test_start:.2f}s")
    return data_dicts
#################################################################################################################################################

if __name__ == '__main__':
    pass
    # pltsetFont(fontsize=12)
    # Trainresult_Plottoshow(path='E:\Code\pyCode\Butterfly_Identify\loss\hist_2024-07-22_00-09-45_lr2e-05bs5.json')
    # a = torch.tensor([1,2,0,6,3,13,5,5,6,7,8,8,8,])
    # b = torch.tensor([1,2,0,1,4,5,6,7,6,6,6,6,9])
    # x = MetricMonitor()
    # x.update(b,a,is_Max=False,updata_recall=True,update_f1=True)
    # print(x._Accuracy_,x._Recallscore_,x._F1score_)
    # a1 = torch.tensor([4,6,2,1,5])
    # b1 = torch.tensor([4,5,2,4,6])
    # x.update(b1,a1,is_Max=False,updata_recall=True,update_f1=True)
    # print(x._Accuracy_,x._Recallscore_,x._F1score_)
    # with open('E:\Code\pyCode\HandWriten\outdata\Run2024-07-31_18-46-06\\result.json','r',encoding='utf-8') as f:
    #     datadict = json.load(f)
    datadict = {'targets': [5, 6, 4, 9, 1, 0, 9, 7, 0, 4, 3, 6, 7, 4, 1, 3], 'predict_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'accuracy': [0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    To_save_csv(datadict=datadict,path='history1.csv')
    # pltsetFont(12)
    # Trainresult_Plottoshow(datadict=datadict,toshow=True)
    a = []
