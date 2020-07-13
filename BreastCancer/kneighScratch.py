import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib import style
import pandas as pd
import warnings
from collections import Counter
import random

#k-neighbors uses euclidean distances to measure distances between the prediction point and the given points

style.use('fivethirtyeight')

data= { 'k' : [[1,3],[2,2],[3,4]] , 'r': [[7,7],[8,9],[6,8]] }

for i in data:
    for j in data[i]:
        plt.scatter(j[0],j[1],color=i)

pred_point= [5,4]

def k_nearest_neighbors(data,predict,k=3):
    if(len(data))>=k:
        warnings.warn("K should be more than length og data set")
    dist=[]
    for i in data:
        for j in data[i]:
            euc_dist=np.linalg.norm(np.array(j)-np.array(predict))
            dist.append([euc_dist,i])
    dist.sort()
    votes=[]
    for i in range(k):
        votes.append(dist[i][1])
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

result= k_nearest_neighbors(data,pred_point,k=3)

plt.scatter(pred_point[0],pred_point[1],color=result,s=50)
plt.show()




        
