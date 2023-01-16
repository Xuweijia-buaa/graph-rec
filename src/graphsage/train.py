from process import*
from utils import*
from model import GraphSAGE
from eval import*

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import json
import gzip
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import dgl.data
import os
datapath='/media/xuweijia/新加卷/data/Amazon Product Data/'
def train(fp=os.path.join(datapath+"meta_Electronics.json.gz"), epochs=100, hit=True):
    '''
    Processes Data & Trains GraphSAGE Model

    parameters: str Filepath to data(fp),  item_df的路径。没用u-i inter
                int # Of Epochs (epochs), 
                boolean Compute Hit-Rate (hit)


    returns: tensor Trained embeddings (h),
            GraphSAGE Trained Model (model)

    '''
    #Preprocess
    edges, node_features, node_labels, edges_src, edges_dst = preprocess(fp)

    #Build DGL Graph
    # 亚马逊原数据中提供的itemdf中，提供了每个item的n个‘also-buy‘item
    # 以此为基础，构建(u, v)同质图。节点是item。边是item(u)对应的also_buy_item(v) （u,v都label_encoder过了，把item_id映射成了node_id）。
    # 有向图，没有反向边
    # 每个item的初始向量，是item的评论特征（tfidf）  ['feat']
    # 每个item的商品类别，作为每个节点的label        ['label']
    g = build_graph(node_features, node_labels, edges_src, edges_dst)

    #Train-Test Split Graphs
    # 按g中总的边切分。随机选10%的边，作为测试边。g中删掉这些边，构建训练子图
    # train_g:  g中删掉test边后剩下的图。节点数不变，同g，只删掉了测试边
    # train_pos_g：只含train中边的子图（节点数不变）
    # train_neg_g：是和train中n条边，对应的n条负边构成的图
    # test_pos_g： 只含test边的子图  （节点数不变）。可用src预测dst
    train_g, train_pos_g, train_neg_g, test_pos_g, _ = train_test_split(g)

    #Inits
    model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
    pred = DotPredictor()  # 用uv的向量内积，算每个pair的score
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    epochs = epochs

    #Training
    for epoch in range(epochs):
        #Forward
        h = model(train_g, train_g.ndata['feat'])   # 整个图一起训练。没有用mini-batch. 每次epoch得到所有节点的表示 （N,d）
                                                    # train_g中不含测试边，但含所有节点。 得到所有节点的表示 （N,h）

        pos_score = pred(train_pos_g, h)           # 传入整个图，算每个pair的i2i score.  (只含正边)   （E_train,1） (用train_g也行，边一样。)
        neg_score = pred(train_neg_g, h)           # 传入整个负图（只含负边）。算每个pair的i2i score    (E_train,1),边数目相同

        loss = compute_loss(pos_score, neg_score)  # point-wise loss。 正负样本不一定一一对应。 正样本label=1,负样本label=0
        
        #Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('In epoch {}, loss: {}'.format(epoch, loss))

    if hit:
        # edges：原始数据中，用来做边的df，含全部i2i的边['asin', 'also_bought']
        # h:    训练结束，g中每个节点的最终表示h （N,d） 用来eval
        # node_features：  每个item的初始向量，是item的评论特征tfidf   (N,d)
        # test_pos_g：     只含test边的子图，可用src预测dst  （节点数不变）
        # model(test_pos_g, node_features)：
        #      输入只含测试边的子图（节点数不变，同g中全部节点）。把对应的特征设置到该子图的节点上。聚合特征到对应边上，
        #      经过massage passing, 把源节点src的特征通过test边，聚合到test边对应的目标item上。
        #      最终得到test边中，每个dst向量的表示 （N_test,d）（待预测节点dst, 只聚合了来自test边的信息，和dst节点本身的初始信息）
        hits = get_hits(edges, h, model(test_pos_g, node_features))

        # 这里直接让待预测的目标节点dst，聚合了src的信息和自己的信息。不太对：

        # 按理说，应该每个src_item,聚合邻居，得到他自己的表示后。再去预测dst. 这时候，应该是src节点去聚合邻居。
        # 如在src作为seeds采样得到的blocks子图中，作为block[-1]的dst节点，得到表示
        #      也可以直接用g中得到的src节点的表示。但信息不全，只聚合了部分邻居。（且推断型的，train完，主要是训练了网络）
        #      只能一般用他的所有邻居，结合网络infer。
        #      因此这里送入的子图，应该是src的邻居子图，和最后一阶邻居的特征。 得到源节点的表示。来预测目标节点

        # 整图的话，应该先分别得到源节点，目标节点的最终表示。输入全图，得到表示后，截取待预测边对应节点的表示
        # （如node_classification.py的inference，先整图infer: model.inference(graph, device, batch_size) 。 再获取测试节点表示pred = pred[nid]    ）
        #  （或者link_pred.py中的infer,先整图infer,再获取测试节点的表示. (目标节点最好是一些candidate)）

        print(np.mean(hits))
    
    return h, model

if __name__ == '__main__':
    train()


