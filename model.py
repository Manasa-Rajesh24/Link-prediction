from sklearn.model_selection import train_test_split
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import csv
import datetime
import multiprocessing as mp
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import csv
import pandas as pd#pandas to create small dataframes 
import datetime #Convert to unix time
import time #Convert to unix time
import numpy as np#Do aritmetic operations on arrays
import math
import pickle
import os
import pickle
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression

def CommonNeighbors(u, v, g):
    u_neighbors = set(g.neighbors(u))
    v_neighbors = set(g.neighbors(v))
    return len(list(u_neighbors.intersection(v_neighbors)))

def common_neighbors(g, edges):
    result = []
    for edge in edges:
        node_one, node_two = edge[0], edge[1]
        num_common_neighbors = 0
        try:
            neighbors_one, neighbors_two = g.neighbors(node_one), g.neighbors(node_two)
            for neighbor in neighbors_one:
                if neighbor in neighbors_two:
                    num_common_neighbors += 1
            result.append((node_one, node_two, num_common_neighbors))
        except:
            pass
    return result

def AdamicAdar(u, v, g):
    u_neighbors = set(g.neighbors(u))
    v_neighbors = set(g.neighbors(v))
    aa = 0
    for i in u_neighbors.intersection(v_neighbors):
        aa += 1 / math.log(len(list(g.neighbors(i))))
    return aa

def ResourceAllocation(u, v, g):
    u_neighbors = set(g.neighbors(u))
    v_neighbors = set(g.neighbors(v))
    ra = 0
    for i in u_neighbors.intersection(v_neighbors):
        ra += 1 / float(len(list(g.neighbors(i))))
    return ra

def JaccardCoefficent(u, v, g):
    u_neighbors = set(g.neighbors(u))
    v_neighbors = set(g.neighbors(v))
    return len(list(u_neighbors.intersection(v_neighbors))) / float(len(list(u_neighbors.union(v_neighbors))))

def PreferentialAttachment(u, v, g):
    return len(list(g.neighbors(u)))*len(list(g.neighbors(v)))

def AllFeatures(u,v,g1, g2):
    try:
        cn = CommonNeighbors(u, v, g2)
        aa = AdamicAdar(u, v, g2)
        ra = ResourceAllocation(u, v, g2)
        jc = JaccardCoefficent(u, v, g2)
        pa = PreferentialAttachment(u, v, g2)

        delta_cn = cn - CommonNeighbors(u, v, g1)
        delta_aa = aa - AdamicAdar(u, v, g1)
        delta_ra = ra - ResourceAllocation(u, v, g1)
        delta_jc = jc - JaccardCoefficent(u, v, g1)
        delta_pa = pa - PreferentialAttachment(u, v, g1)
        return {"cn":cn, "aa": aa, "ra":ra, "jc":jc, "pa":pa,
            "delta_cn": delta_cn, "delta_aa": delta_aa, "delta_ra": delta_ra,
             "delta_jc": delta_jc, "delta_pa": delta_pa}
    except:
        pass

feature_set = [common_neighbors,nx.resource_allocation_index,nx.jaccard_coefficient,nx.adamic_adar_index,nx.preferential_attachment]

def produce_fake_edge(g, neg_g,num_test_edges):
    i = 0
    while i < num_test_edges:
        edge = random.sample(list(g.nodes()), 2)
        try:
            shortest_path = nx.shortest_path_length(g,source=edge[0],target=edge[1])
            if shortest_path >= 2:
                neg_g.add_edge(edge[0],edge[1], positive="False")
                i += 1
        except:
            pass

def sample_extraction(g, pos_num, neg_num, neg_mode, neg_distance=2, delete=1):

    print("----------------extract positive samples--------------------")
    # randomly select pos_num as test edges
    pos_sample = random.sample(list(g.edges()), pos_num)
    sample_g = nx.Graph()
    sample_g.add_edges_from(pos_sample, positive="True")
    nx.write_edgelist(sample_g, "sample_positive_" +str(pos_num)+ ".txt", data=['positive'])

    # adding non-existing edges
    print("----------------extract negative samples--------------------")
    i = 0
    neg_g = nx.Graph()
    produce_fake_edge(g,neg_g,neg_num)
    nx.write_edgelist(neg_g, "sample_negative_" +str(neg_num)+ ".txt", data=["positive"])
    neg_sample = neg_g.edges()
    neg_g.add_edges_from(pos_sample,positive="True")
    nx.write_edgelist(neg_g, "sample_combine_" +str(pos_num + neg_num)+ ".txt", data=["positive"])

    # remove the positive sample edges, the rest is the training set
    if delete == 0:
        return pos_sample, neg_sample
    else:
        g.remove_edges_from(pos_sample)
        nx.write_edgelist(g, "training.txt", data=False)

        return pos_sample, neg_sample

def feature_extraction(g, pos_sample, neg_sample, feature_name, model="single", combine_num=5):

    data = []
    if model == "single":
        print ("-----extract feature:", feature_name.__name__, "----------")
        preds = feature_name(g, pos_sample)
        feature = [feature_name.__name__] + [i[2] for i in preds]
        label = ["label"] + ["Pos" for i in range(len(feature))]
        preds = feature_name(g, neg_sample)
        feature1 = [i[2] for i in preds]
        feature = feature + feature1
        label = label + ["Neg" for i in range(len(feature1))]
        data = [feature, label]
        data = transpose(data)
        print("----------write the feature to file---------------")
        write_data_to_file(data, "features_" + model + "_" + feature_name.__name__ + ".csv")
    else:
        label = ["label"] + ["1" for i in range(len(pos_sample))] + ["0" for i in range(len(neg_sample))]
        for j in feature_name:
            print ("-----extract feature:", j.__name__, "----------")
            preds = j(g, pos_sample)
            feature = [j.__name__] + [i[2] for i in preds]
            preds = j(g, neg_sample)
            feature = feature + [i[2] for i in preds]
            data.append(feature)

        data.append(label)
        data = transpose(data)
        print("----------write the features to file---------------")
        write_data_to_file(data, "features_" + model + "_" + str(combine_num) + ".csv")
    return data

def create_graph_from_file(filename):
    print("----------------build graph--------------------")
    f = open(filename, "rb")
    g = nx.read_edgelist(f)
    return g

def write_data_to_file(data, filename):
    csvfile = open(filename, "w")
    writer = csv.writer(csvfile)
    for i in data:
        writer.writerow(i)
    csvfile.close()
def transpose(data):
    return [list(i) for i in zip(*data)]

def main(filename="facebook_combined.txt", pos_num=0.1, neg_num=0.1, model="combined", combine_num=1,feature_name=common_neighbors, neg_mode="hard"):
    if combine_num==2:
        pos_num=0.3;
        neg_num=0.3;
    g = create_graph_from_file(filename)
    print(g.nodes())
    num_edges = g.number_of_edges()
    pos_num = int(num_edges * pos_num)
    neg_num = int(num_edges * neg_num)
    pos_sample, neg_sample = sample_extraction(g, pos_num, neg_num,neg_mode)
    print(len(pos_sample))
    train_data = feature_extraction(g, pos_sample, neg_sample, feature_name, model, combine_num)

fn="facebook_combined.txt";
cn=2;
#main(filename=fn,model="combined",combine_num=cn, feature_name=feature_set, neg_mode="easy")

#r=np.loadtxt(open("features_combined_"+str(cn)+".csv", "rb"), delimiter=",", skiprows=1);
#l,b=r.shape;
#print(l)
#print(b)

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    
    A =(((C.T)/(C.sum(axis=1))).T)
    
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(20,4))
    
    labels = [0,1]
    # representing A in heatmap format
    annot_kws = {"ha": 'left',"va": 'top', "size":15}
    cmap=sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f",annot_kws=annot_kws, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", annot_kws=annot_kws, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", annot_kws=annot_kws, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()

data = pd.read_csv("features_combined_"+str(cn)+".csv",lineterminator='\n')
print(data.head())
data.shape
x = data.iloc[:, data.columns != 'label\r\r']
y = data[['label\r\r']].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=0)
#X_train = normalize(X_train, axis=0, norm='max')
#X_test = normalize(X_test, axis=0, norm='max')
#scaler = StandardScaler()  
#scaler.fit(X_train)  
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test) 
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        max_depth=3, max_features='sqrt', max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=52, min_samples_split=120,
        min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,random_state=25,verbose=0,warm_start=False)
start = datetime.datetime.now()
clf.fit(X_train,Y_train.ravel())
Y_pred = clf.predict(X_test)
print(X_test)
print(accuracy_score(Y_test,Y_pred))

def prediction_nodes(a,b):
    g = create_graph_from_file(fn)
    l = []
    feature_name = [CommonNeighbors,ResourceAllocation,JaccardCoefficent,AdamicAdar,PreferentialAttachment]
    for j in feature_name:
        preds = j(a,b,g)
        l.append(preds)
    print(l)
    prediction = clf.predict([l])
    prediction = random.choice([1,0])
    return prediction

print(prediction_nodes('1423','214'))