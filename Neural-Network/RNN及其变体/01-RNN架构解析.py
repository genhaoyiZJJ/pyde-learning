# -*- coding: utf-8 -*-
# @Author  : Chinesejun
# @Email   : itcast@163.com
# @File    : 01-RNN架构解析.py
# @Software: PyCharm

# todo 1:传统RNN模型
# ======
# 导入工具包
import torch
import torch.nn as nn
import torch.nn.functional as F

# # todo 1：RNN模型
# '''
# 第一个参数：input_size(输入张量x的维度)
# 第二个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
# 第三个参数：num_layer(隐藏层的数量)
# '''
# rnn = nn.RNN(5, 6, 3) #A
# '''
# 第一个参数：sequence_length(输入序列的长度)
# 第二个参数：batch_size(批次的样本数量)
# 第三个参数：input_size(输入张量的维度)
# '''
# input = torch.randn(2, 3, 5) #B
# '''
# 第一个参数：num_layer * num_directions(层数*网络方向)
# 第二个参数：batch_size(批次的样本数)
# 第三个参数：hidden_size(隐藏层的维度， 隐藏层神经元的个数)
# '''
# h0 = torch.randn(3, 3, 6) #C
# output, hn = rnn(input, h0)
# # print(output)
# print('outputshape==',output.shape) #torch.Size([1, 3, 6])  1--》 seq_length 3 --batch_size 6 -- 隐藏层节点
# # print(hn)
# print('hnshape===',hn.shape) # torch.Size([1, 3, 6])
# # print('nn.shape===', rnn)

# 注意点： 上面中的三个一的意义不一样 A和C的一需要保持一致， 也可以自己指定
# B中的1可以自己任意改写， 因为是程序员自己指定，

# todo 2:LSTM模型
# 定义LSTM的参数含义: (input_size, hidden_size, num_layers)
# 定义输入张量的参数含义: (sequence_length, batch_size, input_size)
# 定义隐藏层初始张量和细胞初始状态张量的参数含义:
# (num_layers * num_directions, batch_size, hidden_size)
#
# '''
# 第一个参数：input_size(输入张量x的维度)
# 第二个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
# 第三个参数：num_layer(隐藏层层数)
# '''
# rnn = nn.LSTM(5, 6, 2) # bidirectional=True 设置双向
# '''
# 第一个参数：sequence_length(输入序列的长度)
# 第二个参数：batch_size(批次的样本数量)
# 第三个参数：input_size(输入张量x的维度)
# '''
# input = torch.randn(1, 3, 5)
# '''
# 第一个参数：num_layer * num_directions(隐藏层层数*方向数)
# 第二个参数：batch_size(批次的样本数量)
# 第三个参数：hidden_size(隐藏层的维度)
# '''
# h0 = torch.randn(2, 3, 6)
# c0 = torch.randn(2, 3, 6)
# # 将input1,  h0, c0输入到lstm中， 输出结果
# output, (hn, cn) = rnn(input, (h0, c0))
# # print(output)
# print('lstmoutput===',output.shape)
# # print(hn)
# print('lstmhn===',hn.shape)
# # print(cn)
# print('lstmcn===',cn.shape)

# todo 3:GRU模型
'''
第一个参数：input_size(输入张量x的维度)
第二个参数：hidden_size(隐藏层的维度， 隐藏层神经元的个数)
第三个参数：num_layers(隐藏层的层数)
'''
# rnn = nn.GRU(5, 6, 1) # bidirectional=True双向的
# '''
# 第一个参数：sequence_length(输入序列的长度)
# 第二个参数：batch_siz(批次样本的数量)
# 第三个参数：input_size(输入张量x的维度)
# '''
# input = torch.randn(1, 3, 5)
# '''
# 第一个参数：num_layers * num_directions(隐藏层的层数 * 方向数)
# 第二个参数：batch_size(批次样本的数量)
# 第三个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
# '''
# h0 = torch.randn(1, 3, 6)
#
# output, hn = rnn(input, h0)
# print(output)
# print(hn)


# todo 4：注意力机制

# # bmm: batch*matrix*matrix
# input = torch.randn(10, 3, 4)
# mat2 = torch.randn(10, 4, 5)
# res = torch.bmm(input, mat2)
# # print(res.size())

class Attn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        """初始化函数中的参数有5个, query_size代表query的最后一维大小
           key_size代表key的最后一维大小, value_size1代表value的导数第二维大小,
           value = (1, value_size1, value_size2)
           value_size2代表value的倒数第一维大小, output_size输出的最后一维大小"""
        super(Attn, self).__init__()
        # 将以下参数传入类中
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 初始化注意力机制实现第一步中需要的线性层.
        self.attn = nn.Linear(self.query_size + self.key_size, value_size1)
        # 初始化注意力机制实现第三步中需要的线性层.
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, output_size)

    def forward(self, Q, K, V):
        """forward函数的输入参数有三个, 分别是Q, K, V, 根据模型训练常识, 输入给Attion机制的
           张量一般情况都是三维张量, 因此这里也假设Q, K, V都是三维张量"""

        # 第一步, 按照计算规则进行计算,
        # 我们采用常见的第一种计算规则
        # 将Q，K进行纵轴拼接, 做一次线性变化, 最后使用softmax处理获得结果
        print('catshape===', torch.cat((Q[0], K[0]), 1).shape)
        print('attnshape===', self.attn(torch.cat((Q[0], K[0]), 1)).shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((Q[0], K[0]), 1)), dim=1)
        print('attn_weight===', attn_weights.shape)

        # 然后进行第一步的后半部分, 将得到的权重矩阵与V做矩阵乘法计算,
        # 当二者都是三维张量且第一维代表为batch条数时, 则做bmm运算
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)
        print('attn_applied==', attn_applied.shape)

        # 之后进行第二步, 通过取[0]是用来降维, 根据第一步采用的计算方法,
        # 需要将Q与第一步的计算结果再进行拼接
        output = torch.cat((Q[0], attn_applied[0]), 1)
        print('outputshape===', output.shape)

        # 最后是第三步, 使用线性层作用在第三步的结果上做一个线性变换并扩展维度，得到输出
        # 因为要保证输出也是三维张量, 因此使用unsqueeze(0)扩展维度
        output = self.attn_combine(output).unsqueeze(0)
        return output, attn_weights

# 调用验证
query_size = 32
key_size = 32
value_size1 =32
value_size2 = 64
output_size = 64
attn = Attn(query_size, key_size, value_size1, value_size2, output_size)
Q = torch.randn(1,1,32)
K = torch.randn(1,1,32)
V = torch.randn(1,32,64)
out = attn(Q, K ,V)
print(out[0].shape)
print(out[1].shape)
'''
catshape=== torch.Size([1, 64])
attnshape=== torch.Size([1, 32])
attn_weight=== torch.Size([1, 32])
attn_applied== torch.Size([1, 1, 64])
outputshape=== torch.Size([1, 96])
torch.Size([1, 1, 64])
torch.Size([1, 32])
'''
