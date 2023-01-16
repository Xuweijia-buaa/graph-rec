from dgl.nn.pytorch import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    '''
    DGL 2-Layer GraphSAGE Model: 2层layer的GraphSage
    Source: https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/sageconv.html
    '''
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        # 每次SageConv,h_l -> h_l+1. 对应公式：https://docs.dgl.ai/en/0.4.x/api/python/nn.pytorch.html#sageconv。聚合邻居节点
        # 不采样邻居，直接把全部源节点作为邻居，聚合全部邻居产生的消息。
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean') # 每层graphSage。 指定输入，输出特征的维度。以及聚合方式
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')  # 可选多种聚合方式：dgl.nn.SAGEConv(hid_size, out_size, "gcn")

        # mean: 是源节点的平均+自己。只聚合邻居，最后加自己，相当于resnet。好一些
        #       h_u = W(mean(h_v))+ h_v    :graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
        # gcn:  将邻居连同自己一起平均。最后不加自己了
        #       h_u=  W*(sum(h_v) + h_v)/入度： graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))/ degress
        # pool: 是源节点mlp后，各邻居max pooling,每维取最大信号：
        #       h_u = W* max(Relu(W*h_v))  :graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
    
    def forward(self, g, in_feat):
        '''
        每层之间用了一个Relu.得到最终每个节点的表示。
        -------
        graph:  用来update_all。聚合邻居时，作为消息传递的中介:
                每层的输入特征，被设置到g.srcdata['h']上，产生/更新消息。
                最后拿出每个节点上的聚合结果graph.dstdata['neibor'],作为该层的输出：（N,D）。 是该层每个节点的新表示
        in_feat: 每个节点的初始embedding（N,D）
        '''
        h = self.conv1(g, in_feat)   # infeat (N,in)-> h  (N,out)
        h = F.relu(h)                # 每层之间用了一个Relu.得到最终每个节点的表示
        h = self.conv2(g, h)
        return h                     # 得到最终每个节点的表示