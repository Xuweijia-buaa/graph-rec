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

from utils import getDF

def preprocess(fp):
    '''
    Cleans and process Amazon product metadata

    Parameters: str Filepath (fp)  item_info
    Returns: DataFrame edges, 
            Tensor node_features, 
            Tensor node_labels, 
            Tensor edges_src, 
            Tensor edges_dst
    '''
    # item_df.每行一条item记录
    df = getDF(fp)
    df = df.dropna()
    # item_df中，每个item本身含属性related,作为dict，含‘also_bought’属性   只留有 "Also Bought" 的item（去掉孤立节点）
    df.related = df.related.apply(lambda x: x if 'also_bought' in x.keys() else np.nan) # related,只留了 "Also Bought"属性，其他属性去掉了
    df = df.dropna()
    df['also_bought'] = df.related.apply(lambda x: x['also_bought'])   # 拿出来，作为‘also_bought‘ column.  每个item对应一个list

    #Remove all product IDs (ASINs) outside the dataset
    df['also_bought'] = df['also_bought'].apply(lambda x: set(df.asin).intersection(x))        # 去除also-buy 不在数据集中的孤立item
    df['also_bought'] = df['also_bought'].apply(lambda x: list(x) if len(x) > 0 else np.nan)
    df = df.dropna().reset_index(drop=True)

    #Finds Niche Category of each product
    df['niche'] = df.categories.apply(lambda x: str(x).strip(']').split(',')[-1]) # 商品类别。

    df = df.explode('also_bought')   # 行转列，df每条记录变成ii pair。  ii是i和also-buy的i  （also-buy的列名不变）

    all_nodes = list(set(df.asin).intersection(set(df.also_bought)))  # 交集，作为全部nodes. 只包含i2i里的。不包含没有交互过的item。

    edges = df[['asin', 'also_bought']]   # 用来做边的df   i2i,每个共同购买，作为一条边

    #Map String ASINs (Product IDs) to Int IDs
    asin_map_dict = pd.Series(edges.asin.append(edges.also_bought).unique()).reset_index(drop=True).to_dict()
    asin_map = {v: k for k, v in asin_map_dict.items()}   # {node_id: item_id}

    edges.asin = edges.asin.apply(lambda x: asin_map[x])
    edges.also_bought = edges.also_bought.apply(lambda x: asin_map[x])
    edges = edges.reset_index(drop=True)                  #  df中itemid,映射成node_id:   'asin', 'also_bought'

    #Text Manipulations
    text_df = df[['asin','title','niche']]
    text_df.asin = text_df.asin.apply(lambda x: asin_map[x])            # {item_id:text}
    text_df = text_df.drop_duplicates('asin').reset_index(drop=True)

    #TF-IDF Vectorizer for Title Text Feature
    corpus = list(text_df.title)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)   # 每个item，对应text的|I,d|

    node_features = torch.Tensor(X.toarray())        # 每个item的初始向量，是item的评论特征
    node_labels = torch.from_numpy(text_df['niche'].astype('category').cat.codes.to_numpy())
    edges_src = torch.from_numpy(edges['asin'].to_numpy())          # u    原始item
    edges_dst = torch.from_numpy(edges['also_bought'].to_numpy())   # v    u的also-buy item

    return edges, node_features, node_labels, edges_src, edges_dst

def build_graph(node_features, node_labels, edges_src, edges_dst):
    '''
    Builds DGL Graph
    '''
    #Builds graph
    g = dgl.graph((edges_src, edges_dst))      # (u,v)同质图。节点都是item,边是亚马逊原数据中提供的，item(u)对应的also_buy_item(v)。 有向图
    g.ndata['feat'] = node_features            # 每个item的初始向量，是item的评论特征（tfidf）
    g.ndata['label'] = node_labels             # 每个item的商品类别，作为每个节点的label

    return g

def train_test_split(g):
    '''
    Splits the graph by edges for training and testing

    输入原始g.每条边是有关联的i2i。做link-predict。 因此划分部分边(对应uv),构建test子图。g中删掉这些边

    无监督设置： 原始g中每条边是正向pair.有边相连的ii之间都是co-pruchase。可以看做正边。需要为每条边采样负边
               这里是整图训练。
               所以把整个图中，所有没有连接的边作为负边。采样和原图一样多的边。构建负样本对应的图g-
               图g-中，边的数量和原图g相同，每条边，作为原图g中每条边的负边。u相同，但v不同，是原图g中不存在的边，即和i之前无关联的item
               如uv矩阵。每行可以从值为0的列中，采样u的负样本，如v2。I条边，采样得到的I条负边，构建负图g.
                      v1    v2  v3
                u     1     0   0

                参考：Source: https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
                    每个epoch,为图g生成一个对应的负图-g。对应的正负边，源节点相同，随机采样一些节点作为负边。 每个图分别计算pair-score. 再计算正负pair的max-margin-loss

    Returns:
        train-g: g中删掉test边后剩下的图。节点数不变，同g，只删掉了测试边  （test边是随机选的10%的边）
        train_pos_g：只含train中边的子图（无多余节点）
        train_neg_g：是和train中n条边，对应的n条负边构成的图
        test_pos_g： 只含test边的子图  （无多余节点）。可以用正边src预测dst
        test_neg_g： 和test的n条边，对应的n条负边构成的图（没用）
    '''
    # Split edge set for training and testing
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)

    # 对整个图的边(eid)进行划分。
    # 用10%的边做link-predict测试。训练集中不含。
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]     # 测试集中的边 （对应u,v）
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]   # 训练集中的边 （对应u,v）

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))   # 原始整个图对应的uv
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())        # 1-，得该图中所有没有的边。去掉自连接
    neg_u, neg_v = np.where(adj_neg != 0)                            # 把图中没有的边(对应uv)，作为负样本。 uv从未一起被购买过

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)                # 选和g中边相同数量的负边，作为最终的负边
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]   # 划分到train/test中
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]] # 训练集中的边 （对应u,v）（原始g中没有的边，ii间没有关联）

    train_g = dgl.remove_edges(g, eids[:test_size])     # train_g: 原始g中,去掉测试集中的边。

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())   # 只含train中边
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())   # 只含trian中负边

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())      # 只含test中正边
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())      # 只含test中负边 （没用。直接用正边src预测dst）

    return train_g,\
           train_pos_g, train_neg_g,\
           test_pos_g,test_neg_g