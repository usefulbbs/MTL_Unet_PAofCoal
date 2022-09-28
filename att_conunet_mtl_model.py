import torch
import torch.nn as nn
from shared_model import CNN_model
from torch.nn.modules.loss import MSELoss
from sklearn.metrics import mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):    #定义一个二重卷积模块
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1,stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result







class MMOe(nn.Module):
    def __init__(self,n_expert=1,mmoe_hidden_dim=125,hidden_dim=[256,64],dropouts=[0.1,0.1],hidden_size = 125,num_task=4,
                output_size=1,expert_activation=None,hidden_size_gate=125):
        super(MMOe,self).__init__()
        # # experts layer/shared layer
        # self.experts = CNN_model
        # 定义卷积层池化层
        self.conv1 = DoubleConv(1, 32)
        self.att1 = CBAM(32, 1, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = DoubleConv(32, 64)

        self.upsam1 = nn.ConvTranspose1d(64,32,3, stride=2)

        self.att2 = CBAM(64, 1, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = DoubleConv(64, 128)

        self.upsam2 = nn.ConvTranspose1d(128,32,5,stride=4)

        #self.att3 = CBAM(128, 1, 3)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = DoubleConv(128, 256)

        self.upsam3 = nn.ConvTranspose1d(256,32,3,stride=9,padding=2)

        #self.att4 = CBAM(256, 1, 3)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = DoubleConv(256, 512)

        #self.upsam3 = nn.ConvTranspose1d(512,32,5,stride=20,padding=0)

        self.up6 = nn.ConvTranspose1d(512, 256, 3, stride=2)
        self.conv6 = DoubleConv(512, 256)

        self.upsam4 = nn.ConvTranspose1d(256,32,3,stride=9,padding=2)

        self.up7 = nn.ConvTranspose1d(256, 128, 3, stride=2)
        self.conv7 = DoubleConv(256, 128)

        self.upsam5 = nn.ConvTranspose1d(128,32,5,stride=4)

        self.up8 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose1d(64, 32, 3, stride=2)
        self.conv9 = DoubleConv(128, 32)
        self.last_layer = nn.Conv1d(32, 1, 1)

        self.loss_fun = nn.L1Loss()
        self.num_task = num_task
        self.weights = torch.nn.Parameter(torch.ones(self.num_task).float())

        self.expert_activation = expert_activation
        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)

        # gates
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size_gate, n_expert).to(device), requires_grad=True) for _ in
                      range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert).to(device), requires_grad=True) for _ in
                           range(num_task)]



        # Tower
        for i in range(self.num_task):
            setattr(self,'task_{}_dnn'.format(i+1),nn.ModuleList())
            hid_dim = [mmoe_hidden_dim]+hidden_dim
            for j in range(len(hid_dim)-1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i+1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i+1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], output_size))


    def forward(self,x,y):  #x=[batchsize,125]
        #mmoe
        c1 = self.conv1(x)  ##32x32x125
        c11 = self.att1(c1)
        p1 = self.pool1(c1)   #32x32x62
        #print(p1.shape)
        c2 = self.conv2(p1)   #32x64x62

        part4 = self.upsam1(c2)

        c22 = self.att2(c2)
        p2 = self.pool2(c2)   #32x64x31
        #print(p2.shape)
        c3 = self.conv3(p2)   #32x128x31

        part5 = self.upsam2(c3)

        #c33 = self.att3(c3)
        p3 = self.pool3(c3)   #32x128x15
        #print(p3.shape)
        c4 = self.conv4(p3)  #32x256x15

        part6 = self.upsam3(c4)

        #c44 = self.att4(c4)
        p4 = self.pool4(c4)  #32x256x7
        #print(p4.shape)
        c5 = self.conv5(p4)   #32x512x7

        #part1 = self.upsam3(c5)

        up_6 = self.up6(c5)   #32x32x15
        merge6 = torch.cat([up_6, c4], dim=1)   #拼接 32x512x15
        c6 = self.conv6(merge6)   #32x256x15

        part2 = self.upsam4(c6)

        up_7 = self.up7(c6)        #32x128x31
        merge7 = torch.cat([up_7, c3], dim=1)   #32x256x31
        c7 = self.conv7(merge7)       #32x128x31

        part3 = self.upsam5(c7)

        up_8 = self.up8(c7)          #32x64x62
        merge8 = torch.cat([up_8, c22], dim=1)   #32x128x62
        c8 = self.conv8(merge8)    #32x64x62
        up_9 = self.up9(c8)       #32x32x125
        # merge9 = torch.cat([up_9, c11], dim=1)  #32x64x125

        concate = torch.cat([part2,part3,up_9,c11],dim=1)   # 32x128x125

        c9 = self.conv9(concate)   #32x32x125
        #加上一个注意力机制模块
        x1 = self.last_layer(c9)  # 32x1x125
        x1 = x1.view(x1.size(0), -1)
        # experts_out = torch.einsum('ij, jkl -> ikl', x1, self.experts)  # [64,128,3]
        # experts_out += self.experts_bias
        # if self.expert_activation is not None:  # [32,256,1]
        #     experts_out = self.expert_activation(experts_out)  # 共享层的最后一层输出

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            shared_output = x1
            # shared_output = torch.squeeze(shared_output, dim=2)
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                shared_output = mod(shared_output)
            task_outputs.append(shared_output)


        task_loss = []
        for j in range(self.num_task):
            a = y[:,j]
            a = torch.unsqueeze(a, dim=1)
            task_loss.append(self.loss_fun(a,task_outputs[j]))
        task_loss =torch.stack(task_loss)

        weighted_task_loss = torch.mul(self.weights,task_loss)


        return weighted_task_loss,task_loss,task_outputs

    def get_last_shared_layer(self):
        return self.last_layer