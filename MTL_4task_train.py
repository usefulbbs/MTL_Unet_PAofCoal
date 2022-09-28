import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from att_conunet_mtl_model import MMOe
from sklearn.metrics import mean_absolute_error,mean_squared_error
import random
import math
import matplotlib.pyplot as plt
import copy
import os
import scipy.io as scio
from sklearn.metrics import mean_absolute_error,mean_squared_error
import argparse



parser = argparse.ArgumentParser(description='GradNorm')
parser.add_argument('--n-iter','-it',type=int,default=70)
parser.add_argument('--mode','-m',choices=('grad_norm', 'equal_weight'), default='grad_norm')
parser.add_argument('--alpha', '-a', type=float, default=0.12)
parser.add_argument('--sigma', '-s', type=float, default=100.0)
args = parser.parse_args()











rand_seed = 64678
if rand_seed is not None:
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq

    return corr_factor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#预处理2st+SG
dataFile_train_x='E:/小论文资料/程序/CARS/New Folder/MTL_train_data1.mat'
dataFile_train_y='E:/小论文资料/程序/CARS/New Folder/MTL/MTL_531_label_spxy.mat'
dataFile_test_x='E:/小论文资料/程序/CARS/New Folder/MTL_test_data1.mat'
dataFile_test_y='E:/小论文资料/程序/CARS/New Folder/MTL/MTL_133_label_spxy.mat'

Y_data = scio.loadmat(dataFile_train_y)
X_data = scio.loadmat(dataFile_train_x)
X_test = scio.loadmat(dataFile_test_x)
Y_test = scio.loadmat(dataFile_test_y)


print(X_data['train_data'].shape)
X_data = X_data['train_data']
X_data = pd.DataFrame(X_data)

Y_data = Y_data['MTL_531_label_spxy']
Y_data = pd.DataFrame(Y_data)

X_test = X_test['test_data']
X_test = pd.DataFrame(X_test)

Y_test = Y_test['MTL_133_label_spxy']
Y_test = pd.DataFrame(Y_test)




num_task = 4
Y_data = Y_data.values
Y_test = Y_test.values
# Y_data = Y_data[:,3:4]
# Y_test = Y_test[:,3:4]
Y_Mt_data = Y_data[:,0]
Y_Mt_test = Y_test[:,0]
Y_Aar_data = Y_data[:,1]
Y_Aar_test = Y_test[:,1]

Y_Vd_data = Y_data[:,3]
Y_Vd_test = Y_test[:,3]
Y_Q_data = Y_data[:,4]
Y_Q_test = Y_test[:,4]
print(X_data.shape)
print(Y_Vd_data.shape,Y_Vd_test.shape)
print(X_test.shape)
print(Y_Q_data.shape,Y_Q_test.shape)
#数据标准化处理
scale = StandardScaler()
X_data_s = scale.fit_transform(X_data)
X_test_s = scale.fit_transform(X_test)

#将数据转化成张量
X_data_t = torch.from_numpy(X_data_s.astype(np.float32))
X_test_t = torch.from_numpy(X_test_s.astype(np.float32))
Y_Mt_data = torch.from_numpy(Y_Mt_data.astype(np.float32))
Y_Mt_test = torch.from_numpy(Y_Mt_test.astype(np.float32))
Y_Aar_data = torch.from_numpy(Y_Aar_data.astype(np.float32))
Y_Aar_test = torch.from_numpy(Y_Aar_test.astype(np.float32))

Y_Vd_data = torch.from_numpy(Y_Vd_data.astype(np.float32))
Y_Vd_test = torch.from_numpy(Y_Vd_test.astype(np.float32))
Y_Q_data = torch.from_numpy(Y_Q_data.astype(np.float32))
Y_Q_test = torch.from_numpy(Y_Q_test.astype(np.float32))

Y_Mt_data = torch.unsqueeze(Y_Mt_data, dim=1)
Y_Mt_test = torch.unsqueeze(Y_Mt_test, dim=1)
Y_Aar_data = torch.unsqueeze(Y_Aar_data, dim=1)
Y_Aar_test = torch.unsqueeze(Y_Aar_test, dim=1)

Y_Vd_data = torch.unsqueeze(Y_Vd_data, dim=1)
Y_Vd_test = torch.unsqueeze(Y_Vd_test, dim=1)
Y_Q_data = torch.unsqueeze(Y_Q_data, dim=1)
Y_Q_test = torch.unsqueeze(Y_Q_test, dim=1)


Y_test = torch.cat((Y_Mt_test,Y_Aar_test,Y_Vd_test, Y_Q_test), 1)
print(X_data_t.shape,X_test_t.shape,Y_Vd_data.shape,Y_Vd_test.shape)


#交叉验证
kf = KFold(n_splits=6, shuffle=True, random_state=1)
save_path = 'mtl_weigth/MMOE.pth'
loss_test_mt = []
loss_test_aar = []
loss_test_vd=[]
loss_test_q=[]
r_test_mt = []
r_test_aar = []
r_test_vd = []
r_test_q = []
rmse_test_mt = []
rmse_test_aar = []
rmse_test_vd = []
rmse_test_q = []
for k, (train_index, test_index) in enumerate(kf.split(X_data_t)):  #交叉验证划分训练集和验证集
    X_train = X_data_t[train_index]
    x_test = X_data_t[test_index]
    Y_Mt_train = Y_Mt_data[train_index]
    y_Mt_test = Y_Mt_data[test_index]
    Y_Aar_train = Y_Aar_data[train_index]
    y_Aar_test = Y_Aar_data[test_index]

    Y_Vd_train = Y_Vd_data[train_index]
    y_Vd_test = Y_Vd_data[test_index]
    Y_Q_train = Y_Q_data[train_index]
    y_Q_test = Y_Q_data[test_index]
    Y_train = torch.cat((Y_Mt_train,Y_Aar_train,Y_Vd_train,Y_Q_train),1)
    y_test = torch.cat((y_Mt_test,y_Aar_test,y_Vd_test,y_Q_test),1)
    # 再对训练集划分小训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=20)
    # 定义数据加载器
    train_data = Data.TensorDataset(x_train, y_train)
    val_data = Data.TensorDataset(x_val, y_val)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=31, shuffle=True, num_workers=0)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=31, shuffle=True, num_workers=0)
    # 超参
    # learn_rate = 0.001
    loss_fun = nn.L1Loss()
    # mmoe
    mmoe = MMOe(num_task = 4)
    optimizer = torch.optim.Adam(mmoe.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.67)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # optimizer = torch.optim.SGD(mmoe.parameters(),lr=learn_rate)
    mmoe.to(device)
    patience,eval_loss = 0,0
    min_loss = 1000

    Training_R_all_Mt = []
    Training_R_all_Aar = []

    Training_R_all_Vd = []
    Training_R_all_Q = []
    Training_loss_all = []

    Validate_R_all_Mt = []
    Validate_R_all_Aar = []

    Validate_R_all_Vd = []
    Validate_R_all_Q = []

    Validate_loss_all = []
    weights = []  #每个任务的损失权重
    task_losses = []  #每个任务的损失列表
    loss_ratios = []    # 任务的相对反向训练速度
    grad_norm_losses = []    # 每个任务损失加权之和
    #train
    for i in range(args.n_iter):
        #train
        mmoe.train() #训练过程使用dropout和bn
        total_loss, R_Mt_train, R_Aar_train, R_Vd_train,R_Q_train,count = 0, 0, 0, 0, 0, 0
        for idx,(b_x,b_y) in enumerate(train_loader):
            b_y_Mt1 = b_y[:,0]
            b_y_Aar1 = b_y[:,1]

            b_y_Vd1 = b_y[:,2]
            b_y_Q1 = b_y[:,3]
            b_x,b_y_Mt,b_y_Aar,b_y_Vd,b_y_Q,b_y=b_x.to(device),b_y_Mt1.to(device),b_y_Aar1.to(device),b_y_Vd1.to(device),b_y_Q1.to(device),b_y.to(device)
            b_x = torch.unsqueeze(b_x, dim=1)
            weighted_task_loss,task_loss,predict = mmoe(b_x,b_y)
            predict_Mt = predict[0]
            predict_Aar = predict[1]

            predict_Vd = predict[2]
            predict_Q = predict[3]
            R_Mt = calc_corr(predict_Mt,b_y_Mt)
            R_Aar = calc_corr(predict_Aar,b_y_Aar)

            R_Vd = calc_corr(predict_Vd,b_y_Vd)
            R_Q = calc_corr(predict_Q,b_y_Q)
            # b_y_Vd = torch.unsqueeze(b_y_Vd, dim=1)
            # b_y_Q = torch.unsqueeze(b_y_Q, dim=1)
            #
            # loss_Vd = torch.tensor([loss_fun(predict_Vd,b_y_Vd)]).to(device)
            # loss_Q = torch.tensor([loss_fun(predict_Q,b_y_Q)]).to(device)

            # weighted_task_loss = torch.mul(mmoe.weights,task_loss)
            if i == 0:
                # set L(0)
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu()
                else:
                    initial_task_loss = task_loss.data
                initial_task_loss = initial_task_loss.numpy()

            # total loss
            loss1 = torch.sum(task_loss)
            loss = torch.sum(weighted_task_loss)
            #clear the gradients
            optimizer.zero_grad()
            # 计算梯度
            loss.backward(retain_graph=True)

            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            mmoe.weights.grad.data = mmoe.weights.grad.data * 0.0

            if args.mode == 'grad_norm':
                # 得到共享层输出权重
                W = mmoe.get_last_shared_layer()
                # 对每个任务计算基于该权重的梯度L2范数
                # G^{(i)}_w(t)
                norms = []
                for j in range(len(task_loss)):
                    # 任务j的带权损失 对需要更新的共享权重W的梯度
                    gLgW = torch.autograd.grad(task_loss[j], W.parameters(),retain_graph=True)
                    # 该梯度值的L2范数
                    norms.append(torch.norm(torch.mul(mmoe.weights[j],gLgW[0])))
                norms = torch.stack(norms)  # j个任务每个损失乘以权重后的梯度的L2范数：G^{(i)}_w(t)

                # 计算反向学习速率r_i(t)
                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)

                # 计算每个任务G^{(i)}_w(t)的均值
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())

                # 计算Gradnorm Loss
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()
                grad_norm_loss = torch.sum(torch.abs(norms-constant_term)).requires_grad_(True)
                # 计算grad loss gradient
                grad_norm_loss = grad_norm_loss.requires_grad_(requires_grad=True)
                mmoe.weights.grad = torch.autograd.grad(grad_norm_loss,mmoe.weights)[0]

            # 整个网络权重的优化
            optimizer.step()
            total_loss += loss1.item() * b_x.size(0)
            R_Mt_train += R_Mt.item() * b_x.size(0)
            R_Aar_train += R_Aar.item() * b_x.size(0)

            R_Vd_train += R_Vd.item() * b_x.size(0)
            R_Q_train += R_Q.item() * b_x.size(0)
            count += b_x.size(0)
            # renormalize  下一个batch使用的是renormalize之后的w
            normalize_coeff = num_task / torch.sum(mmoe.weights.data, dim=0)
            mmoe.weights.data = mmoe.weights.data * normalize_coeff

            # record
            if torch.cuda.is_available():
                task_losses.append(task_loss.data.cpu().numpy())
                loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                weights.append(mmoe.weights.data.cpu().numpy())
                grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
            else:
                task_losses.append(task_loss.data.numpy())
                loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                weights.append(mmoe.weights.data.numpy())
                grad_norm_losses.append(grad_norm_loss.data.numpy())

            # if torch.cuda.is_available():
            #     print('{}/{}：loss_ratio={},weights={},task_loss={},grad_norm_loss={}'.format(
            #             i,args.n_iter,loss_ratios[-1], mmoe.weights.data.cpu().numpy(), task_loss.data.cpu().numpy(), grad_norm_loss.data.cpu().numpy()))
            # else:
            #     print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
            #         i, args.n_iter, loss_ratios[-1], mmoe.weights.data.numpy(), task_loss.data.numpy(), grad_norm_loss.data.numpy()))
        scheduler.step()
        print("Epoch %d train loss is %.4f, Mt_R_train is %.4f, Aar_R_train id %.4f, Vd_R_train is %.4f , Q_R_train is %.4f" % (i + 1, total_loss / count,R_Mt_train / count ,R_Aar_train / count,R_Vd_train / count, R_Q_train / count))

        # 验证
        mmoe.eval()
        total_val_loss,count_val ,R_Mt_val,R_Aar_val,R_Sar_val,R_Vd_val,R_Q_val= 0,0,0,0,0,0,0
        for idx,(b_x,b_y) in enumerate(val_loader):
            b_y_Mt1 = b_y[:, 0]
            b_y_Aar1 = b_y[:, 1]

            b_y_Vd1 = b_y[:, 2]
            b_y_Q1 = b_y[:, 3]
            b_x, b_y_Mt, b_y_Aar, b_y_Vd, b_y_Q, b_y = b_x.to(device), b_y_Mt1.to(device), b_y_Aar1.to(device), b_y_Vd1.to(device), b_y_Q1.to(device), b_y.to(device)
            b_x = torch.unsqueeze(b_x, dim=1)
            weighted_task_loss,task_loss,predict = mmoe(b_x,b_y)
            predict_Mt = predict[0]
            predict_Aar = predict[1]

            predict_Vd = predict[2]
            predict_Q = predict[3]
            R_Mt = calc_corr(predict_Mt,b_y_Mt)
            R_Aar = calc_corr(predict_Aar,b_y_Aar)

            R_Vd = calc_corr(predict_Vd,b_y_Vd)
            R_Q = calc_corr(predict_Q,b_y_Q)
            b_y_Mt = torch.unsqueeze(b_y_Mt,dim=1)
            b_y_Aar = torch.unsqueeze(b_y_Aar,dim=1)

            b_y_Vd = torch.unsqueeze(b_y_Vd, dim=1)
            b_y_Q = torch.unsqueeze(b_y_Q, dim=1)
            loss_Mt_val = loss_fun(predict_Mt,b_y_Mt)
            loss_Aar_val = loss_fun(predict_Aar,b_y_Aar)

            loss_Vd_val = loss_fun(predict_Vd,b_y_Vd)
            loss_Q_val = loss_fun(predict_Q,b_y_Q)
            loss_val = loss_Mt_val+loss_Aar_val+loss_Vd_val+loss_Q_val
            total_val_loss += loss_val.item() * b_x.size(0)
            R_Mt_val += R_Mt.item() * b_x.size(0)
            R_Aar_val += R_Aar.item() * b_x.size(0)

            R_Vd_val += R_Vd.item() * b_x.size(0)
            R_Q_val += R_Q.item() * b_x.size(0)
            count_val += b_x.size(0)
        print("Epoch %d val loss is %.4f, Mt_R_val is %.4f,Aar_R_val is %.4f,Vd_R_val is %.4f , Q_R_val is %.4f" % (i + 1, total_val_loss / count_val,
                                                                                     R_Mt_val / count_val,R_Aar_val / count_val,R_Vd_val / count_val, R_Q_val / count_val))
        if i>4:
            Training_loss_all.append(total_loss / count)
            Validate_loss_all.append(total_val_loss / count_val)
        Training_R_all_Mt.append(R_Mt_train / count)
        Training_R_all_Aar.append(R_Aar_train / count)
        Training_R_all_Vd.append(R_Vd_train / count)
        Training_R_all_Q.append(R_Q_train / count)

        Validate_R_all_Mt.append(R_Mt_val / count_val)
        Validate_R_all_Aar.append(R_Aar_val / count_val)
        Validate_R_all_Vd.append(R_Vd_val / count_val)
        Validate_R_all_Q.append(R_Q_val / count_val)

        #拷贝模型最低损失下的参数
        if total_loss / count < min_loss:
            min_loss = total_loss / count
            best_model_wts = copy.deepcopy(mmoe.state_dict())
            torch.save(best_model_wts,save_path)



    task_losses = np.array(task_losses)
    weights = np.array(weights)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title(r'Change of weights $w_i$ over step')
    ax1.set_title(r'Change of task loss Aar over step')
    ax2.set_title(r'Change of task loss Q over step')
    ax3.set_title(r'Change of loss ratios over step')
    ax4.set_title(r'Change of grad norm loss over step')
    ax1.plot(task_losses[:, 0])
    ax2.plot(task_losses[:, 1])
    ax3.plot(loss_ratios)
    ax4.plot(grad_norm_losses)
    ax5.plot(weights[:, 0])
    ax5.plot(weights[:, 1])
    ax5.plot(weights[:, 2])
    ax5.plot(weights[:, 3])

    plt.show()




    # 训练和验证过程作图
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 3, 1)
    plt.plot(Training_loss_all, 'r-', label='Train MAE')
    plt.plot(Validate_loss_all, 'b-', label='val MAE')
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.plot(Training_R_all_Mt, 'g-', label='Train R_Mt')
    plt.plot(Validate_R_all_Mt, 'y-', label='Val R_Mt')
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.plot(Training_R_all_Aar, 'g-', label='Train R_Aar')
    plt.plot(Validate_R_all_Aar, 'y-', label='Val R_Aar')
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.plot(Training_R_all_Vd, 'g-', label='Train R_Vd')
    plt.plot(Validate_R_all_Vd, 'y-', label='Val R_Vd')
    plt.legend()
    plt.subplot(2, 3, 5)
    plt.plot(Training_R_all_Q, 'g-', label='Train R_Q')
    plt.plot(Validate_R_all_Q, 'y-', label='Val R_Q')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.show()
    #测试并求六折均值
    mmoe.load_state_dict(torch.load(save_path))
    mmoe.eval()
    x_test,y_Mt_test,y_Aar_test,y_Vd_test,y_Q_test,y_test = x_test.to(device),y_Mt_test.to(device),y_Aar_test.to(device),y_Vd_test.to(device),y_Q_test.to(device),y_test.to(device)
    x_test = torch.unsqueeze(x_test, dim=1)
    weighted_task_loss,task_loss,output_test = mmoe(x_test,y_test)
    test_mt = output_test[0]
    test_aar = output_test[1]

    test_vd = output_test[2]
    test_q = output_test[3]
    loss_mt = loss_fun(test_mt,y_Mt_test)
    loss_aar = loss_fun(test_aar,y_Aar_test)

    loss_vd = loss_fun(test_vd,y_Vd_test)
    loss_q = loss_fun(test_q,y_Q_test)

    rmse_mt = pow(mean_squared_error(test_mt.cpu().detach().numpy(),y_Mt_test.cpu().detach().numpy()),0.5).item()
    rmse_aar = pow(mean_squared_error(test_aar.cpu().detach().numpy(),y_Aar_test.cpu().detach().numpy()),0.5).item()
    rmse_vd = pow(mean_squared_error(test_vd.cpu().detach().numpy(),y_Vd_test.cpu().detach().numpy()),0.5).item()
    rmse_q = pow(mean_squared_error(test_q.cpu().detach().numpy(),y_Q_test.cpu().detach().numpy()),0.5).item()

    loss_mt = loss_mt.item()
    loss_aar = loss_aar.item()

    loss_vd = loss_vd.item()
    loss_q = loss_q.item()
    r_mt=calc_corr(test_mt,y_Mt_test)
    r_aar=calc_corr(test_aar,y_Aar_test)

    r_vd=calc_corr(test_vd,y_Vd_test)
    r_q=calc_corr(test_q,y_Q_test)
    print(' loss_mt=%.4f,loss_aar=%.4f,loss_vd=%.4f,loss_q=%.4f,R_Mt=%.4f,R_Aar=%.4f,R_Vd=%.4f,R_Q=%.4f'%(loss_mt,loss_aar,loss_vd,loss_q,r_mt,r_aar,r_vd,r_q))

    r_mt = r_mt.item()
    r_aar = r_aar.item()
    r_vd = r_vd.item()
    r_q = r_q.item()
    r_test_mt.append(r_mt)
    r_test_aar.append(r_aar)
    r_test_vd.append(r_vd)
    r_test_q.append(r_q)

    rmse_test_mt.append(rmse_mt)
    rmse_test_aar.append(rmse_aar)
    rmse_test_vd.append(rmse_vd)
    rmse_test_q.append(rmse_q)


    loss_test_mt.append(loss_mt)
    loss_test_aar.append(loss_aar)
    loss_test_vd.append(loss_vd)
    loss_test_q.append(loss_q)

print('交叉验证测试集平均loss_mt=%.4f,loss_aar=%.4f,loss_vd=%.4f,loss_q=%.4f'%(np.mean(loss_test_mt),np.mean(loss_test_aar),np.mean(loss_test_vd),np.mean(loss_test_q)))
print('交叉验证测试集平均r_mt=%.4f,r_aar=%.4f,r_vd=%.4f,r_q=%.4f'%(np.mean(r_test_mt),np.mean(r_test_aar),np.mean(r_test_vd),np.mean(r_test_q)))
print('交叉验证测试集平均rmse_mt=%.4f,rmse_aar=%.4f,rmse_vd=%.4f,rmse_q=%.4f'%(np.mean(rmse_test_mt),np.mean(rmse_test_aar),np.mean(rmse_test_vd),np.mean(rmse_test_q)))






rand_seed = 64678
if rand_seed is not None:
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#使用531个训练集进行训练
Y_test = torch.cat((Y_Mt_test,Y_Aar_test,Y_Vd_test, Y_Q_test), 1)
Y_Train = torch.cat((Y_Mt_data,Y_Aar_data,Y_Vd_data, Y_Q_data), 1)
print(X_data_t.shape,Y_Train.shape)
Train_data = Data.TensorDataset(X_data_t, Y_Train)
Train_loader = Data.DataLoader(dataset=Train_data, batch_size=32, shuffle=True, num_workers=0)
model=MMOe()
model.to(device)
loss_function1 = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.67)
save_path_fin = 'mtl_weigth/MMOE_fin.pth'
train_loss_all1 = []
weights = []  # 每个任务的损失权重
task_losses = []  # 每个任务的损失列表
loss_ratios = []  # 任务的相对反向训练速度
grad_norm_losses = []  # 每个任务损失加权之和
for i in range(80):
    train_loss = 0
    total_loss, total_loss1,R_Mt_train,R_Aar_train,R_Vd_train, R_Q_train, count = 0, 0, 0, 0, 0, 0 , 0
    min_loss1 = 1000
    for idx, (b_x, b_y) in enumerate(Train_loader):
        b_y_Mt1 = b_y[:, 0]
        b_y_Aar1 = b_y[:, 1]

        b_y_Vd1 = b_y[:, 2]
        b_y_Q1 = b_y[:, 3]
        b_x, b_y_Mt, b_y_Aar,b_y_Vd, b_y_Q, b_y = b_x.to(device), b_y_Mt1.to(device), b_y_Aar1.to(device),b_y_Vd1.to(device), b_y_Q1.to(device), b_y.to(device)
        b_x = torch.unsqueeze(b_x, dim=1)
        weighted_task_loss, task_loss, predict = model(b_x, b_y)
        predict_Mt = predict[0]
        predict_Aar = predict[1]

        predict_Vd = predict[2]
        predict_Q = predict[3]

        # R_Vd = calc_corr(predict_Vd, b_y_Vd)
        # R_Q = calc_corr(predict_Q, b_y_Q)
        # b_y_Vd = torch.unsqueeze(b_y_Vd, dim=1)
        # b_y_Q = torch.unsqueeze(b_y_Q, dim=1)
        #
        # loss_Vd = torch.tensor([loss_fun(predict_Vd,b_y_Vd)]).to(device)
        # loss_Q = torch.tensor([loss_fun(predict_Q,b_y_Q)]).to(device)

        # weighted_task_loss = torch.mul(mmoe.weights,task_loss)
        if i == 0:
            # set L(0)
            if torch.cuda.is_available():
                initial_task_loss = task_loss.data.cpu()
            else:
                initial_task_loss = task_loss.data
            initial_task_loss = initial_task_loss.numpy()

        # total loss
        loss1 = torch.sum(task_loss)
        loss = torch.sum(weighted_task_loss)
        # clear the gradients
        optimizer.zero_grad()
        # 计算梯度
        loss.backward(retain_graph=True)

        # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
        model.weights.grad.data = model.weights.grad.data * 0.0

        if args.mode == 'grad_norm':
            # 得到共享层输出权重
            W = model.get_last_shared_layer()
            # 对每个任务计算基于该权重的梯度L2范数
            # G^{(i)}_w(t)
            norms = []
            for j in range(len(task_loss)):
                # 任务j的带权损失 对需要更新的共享权重W的梯度
                gLgW = torch.autograd.grad(task_loss[j], W.parameters(), retain_graph=True)
                # 该梯度值的L2范数
                norms.append(torch.norm(torch.mul(model.weights[j], gLgW[0])))
            norms = torch.stack(norms)  # j个任务每个损失乘以权重后的梯度的L2范数：G^{(i)}_w(t)

            # 计算反向学习速率r_i(t)
            if torch.cuda.is_available():
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            else:
                loss_ratio = task_loss.data.numpy() / initial_task_loss
            # r_i(t)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)

            # 计算每个任务G^{(i)}_w(t)的均值
            if torch.cuda.is_available():
                mean_norm = np.mean(norms.data.cpu().numpy())
            else:
                mean_norm = np.mean(norms.data.numpy())

            # 计算Gradnorm Loss
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
            if torch.cuda.is_available():
                constant_term = constant_term.cuda()
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term)).requires_grad_(True)
            # 计算grad loss gradient
            grad_norm_loss = grad_norm_loss.requires_grad_(requires_grad=True)
            model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]

        # 整个网络权重的优化
        optimizer.step()
        total_loss += loss1.item() * b_x.size(0)
        total_loss1 += loss.item() * b_x.size(0)
        # R_Vd_train += R_Vd.item() * b_x.size(0)
        # R_Q_train += R_Q.item() * b_x.size(0)
        count += b_x.size(0)
        # renormalize  下一个batch使用的是renormalize之后的w
        normalize_coeff = num_task / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        # record
        if torch.cuda.is_available():
            task_losses.append(task_loss.data.cpu().numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.cpu().numpy())
            grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
        else:
            task_losses.append(task_loss.data.numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.numpy())
            grad_norm_losses.append(grad_norm_loss.data.numpy())

    # if torch.cuda.is_available():
    #     print('{}/{}：loss_ratio={},weights={},task_loss={},grad_norm_loss={}'.format(
    #         i, args.n_iter, loss_ratios[-1], mmoe.weights.data.cpu().numpy(), task_loss.data.cpu().numpy(),
    #         grad_norm_loss.data.cpu().numpy()))
    # else:
    #     print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
    #         i, args.n_iter, loss_ratios[-1], mmoe.weights.data.numpy(), task_loss.data.numpy(),
    #         grad_norm_loss.data.numpy()))

    scheduler.step()
    train_loss_all1.append(total_loss1 / count)

    print("Epoch %d train loss is %.3f" % (i + 1, total_loss / count))
    # 拷贝模型最低损失下的参数
    if train_loss_all1[-1] < min_loss1:
        min_loss1 = train_loss_all1[-1]
        best_model_wts1 = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts1, save_path_fin)



model2 = MMOe().to(device)
model2.load_state_dict(torch.load(save_path_fin))
model2.eval()
with torch.no_grad():
    X_test_t = X_test_t.to(device)
    Y_test = Y_test.to(device)
    X_test_t = torch.unsqueeze(X_test_t, dim=1)
    weighted_task_loss, task_loss, output2 = model2(X_test_t,Y_test)
    output2_Mt = output2[0]
    output2_Aar = output2[1]

    output2_Vd = output2[2]
    output2_Q = output2[3]
    output2_Mt=output2_Mt.cpu().numpy()
    Y_Mt_test=Y_Mt_test.numpy()
    output2_Aar=output2_Aar.cpu().numpy()
    Y_Aar_test=Y_Aar_test.numpy()


    output2_Vd=output2_Vd.cpu().numpy()
    Y_Vd_test=Y_Vd_test.numpy()
    output2_Q=output2_Q.cpu().numpy()
    Y_Q_test=Y_Q_test.numpy()
    loss_Mt = mean_absolute_error(output2_Mt,Y_Mt_test)
    loss_Aar = mean_absolute_error(output2_Aar,Y_Aar_test)

    loss_Vd = mean_absolute_error(output2_Vd,Y_Vd_test)
    loss_Q = mean_absolute_error(output2_Q,Y_Q_test)
    RMSE_Mt = pow(mean_squared_error(output2_Mt,Y_Mt_test),0.5)
    RMSE_Aar = pow(mean_squared_error(output2_Aar,Y_Aar_test),0.5)

    RMSE_Vd = pow(mean_squared_error(output2_Vd,Y_Vd_test),0.5)
    RMSE_Q = pow(mean_squared_error(output2_Q,Y_Q_test),0.5)
    R_Mt = calc_corr(output2_Mt,Y_Mt_test)

    R_Aar = calc_corr(output2_Aar,Y_Aar_test)
    R_Vd = calc_corr(output2_Vd,Y_Vd_test)
    R_Q = calc_corr(output2_Q,Y_Q_test)


print('RMSE_Mt=%.4f,MAE_Mt=%.4f,R_Mt=%.4f'%(RMSE_Mt,loss_Mt,R_Mt))
print('RMSE_Aar=%.4f,MAE_Aar=%.4f,R_Aar=%.4f'%(RMSE_Aar,loss_Aar,R_Aar))

print('RMSE_Vd=%.4f,MAE_Vd=%.4f,R_Vd=%.4f'%(RMSE_Vd,loss_Vd,R_Vd))
print('RMSE_Q=%.4f,MAE_Q=%.4f,R_Q=%.4f'%(RMSE_Q,loss_Q,R_Q))


