import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(pos_score, neg_score):
    ''' 
    Computes cross entropy loss on edge features
    
    Source: https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
    '''
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)   # point-wise loss

def get_hits(edges_df, h, h_test):
    '''
    Gets list of hits given the parameters below

    parameters: edges_df:原始数据中，用来做边的df，含全部i2i的边['asin', 'also_bought']
                h: 训练完得到的每个节点的最终表示：（N,d）
                h_test: test边中每个待预测item的表示 （N_test,d）.  （只聚合了src和自己的信息）
    returns: list of hits
    '''
    hits = []
    edges = edges_df
    for i in range(h.shape[0]):  # 每个节点
        true_edges = list(edges[edges.asin == i].also_bought)  # 节点i，真正的目标item们
        dist = torch.cdist(h_test[[i]], h)                     # 目标节点(只聚合了源节点u和自己)和所有embed比较
        top_k = torch.topk(dist, k = 500, largest=False)[1]    # 找最接近的item. (也只能是源节点和自己。不太对)
        hit = 0
        for j in true_edges:
            if j in top_k:
                hit = 1
                break
        hits.append(hit)
    return hits