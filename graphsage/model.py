import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

import scipy.sparse as sparse
from pygsp import graphs, filters, plotting, utils

import os
import pickle as pkl
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
chev_order = 3
thre = 0.001
tau = [5, ]
class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora(num_sample = 10):
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set) # wavelet 基过滤后的列表
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    wave_file = "cora"+"chev_"+str(chev_order)+"sample_"+str(num_sample)+".pkl"
    if os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes,num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1
        
        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau) # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j]>thre:
                    ls.append((j,s[i][j]))
            if len(ls)<num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x:x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
    return feat_data, labels, adj_lists, wave_lists

def run_cora(sample_method, gcn=True):
    tau = [3,]
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists, wave_lists = load_cora(5)
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    if sample_method == "adj":
        enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=gcn, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    if sample_method == "adj":
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.item())

    val_output = graphsage.forward(val) 
    rec = ""
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    rec += "\nValidation F1:"+str(f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))
    rec += "\nAverage batch time:"+ str(np.mean(times))
    record("res.txt", "cora"+sample_method+str(gcn)+rec)
    return f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")

def load_pubmed(num_sample = 10):
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    
    wave_file = "pubmed"+"chev_"+str(chev_order)+"sample_"+str(num_sample)+".pkl"
    if os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes,num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1
        
        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau) # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j]>thre:
                    ls.append((j,s[i][j]))
            if len(ls)<num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x:x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
    return feat_data, labels, adj_lists, wave_lists

def run_pubmed(sample_method, gcn=True):
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists, wave_lists = load_pubmed(10)
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    if sample_method == "adj":
        enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        enc1 = Encoder(features, 500, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    if sample_method == "adj":
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.item())

    val_output = graphsage.forward(val) 
    rec = ""
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    rec += "\nValidation F1:"+str(f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))
    rec += "\nAverage batch time:"+ str(np.mean(times))
    record("res.txt", "pubmed"+sample_method+str(gcn)+rec)
    return f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")

def load_citeseer(num_sample = 5):
    #hardcoded for simplicity...
    num_nodes = 3312
    num_feats = 3703
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("citeseer/citeseer.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set) # wavelet 基过滤后的列表
    with open("citeseer/citeseer.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            if info[0] not in node_map or info[1] not in node_map:
                print(info[0], info[1])
                continue
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    
    wave_file = "citeseer"+"chev_"+str(chev_order)+"sample_"+str(num_sample)+".pkl"
    if os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes,num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1
        
        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau) # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j]>thre:
                    ls.append((j,s[i][j]))
            if len(ls)<num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x:x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
    return feat_data, labels, adj_lists, wave_lists

def run_citeseer(sample_method, gcn=True):
    np.random.seed(1)
    random.seed(1)
    num_nodes = 3312
    feat_data, labels, adj_lists, wave_lists = load_citeseer(3)
    features = nn.Embedding(3312, 3703)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    if sample_method == "adj":
        enc1 = Encoder(features, 3703, 128, adj_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        enc1 = Encoder(features, 3703, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    if sample_method == "adj":
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    enc1.num_samples = 3
    enc2.num_samples = 3

    graphsage = SupervisedGraphSage(6, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.item())

    val_output = graphsage.forward(val) 
    rec = ""
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    rec += "\nValidation F1:"+str(f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))
    rec += "\nAverage batch time:"+ str(np.mean(times))
    record("res1.txt", "\nciteseer"+sample_method+str(gcn)+rec)
    return f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")

def record(filename,content):
    with open(filename, 'a') as f:
        f.writelines(content)

if __name__ == "__main__":
    for i in ["wave",]:
        for j in [True, False]:
            avg = 0.0
            for k in range(10): 
                avg+=run_cora(i, j)
            avg/=10
            record("avg.txt","\ncora"+i+str(j)+":"+str(avg))

            # avg = 0.0
            # for k in range(10): 
            #     avg+=run_pubmed(i, j)
            # avg/=10
            # record("avg.txt","\npubmed"+i+str(j)+":"+str(avg))
            
            # avg = 0.0
            # for k in range(10): 
            #     avg+=run_citeseer(i, j)
            # avg/=10
            # record("avg.txt","\nciteseer"+i+str(j)+":"+str(avg))

    # run_cora("adj")
    # run_cora("wave")
    # run_pubmed("adj")
    # run_pubmed("wave")
    # run_citeseer("adj")
    # run_citeseer("wave")
