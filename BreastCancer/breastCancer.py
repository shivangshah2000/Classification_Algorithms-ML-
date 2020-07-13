import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import random

def k_nearest_neighbors(data,predict,k=15):
    if(len(data))>=k:
        warnings.warn("K should be more than length og data set")
    dist=[]
    for i in data:
        for j in data[i]:
            euc_dist=np.linalg.norm(np.array(j)-np.array(predict))  #algo to calculate euclidean dist
            dist.append([euc_dist,i]) #append the dist to dist list
    dist.sort()
    votes=[]
    for i in range(k):
        votes.append(dist[i][1])
  
    vote_result = Counter(votes).most_common(2)[0][0]

    return vote_result


df=pd.read_csv('breast-cancer-wisconsin.data')    #read the data file using pandas dataframe

df.replace('?',-99999,inplace=True) #replace all the missing values with -99999 value

df.drop(['id'],axis=1,inplace=True)  #useless feature 

full_data= df.astype(float).values.tolist() #converts the dataframe into a list of lists easy to work with

random.shuffle(full_data)  #shuffle for better results

test_size=0.2   #80% training data and 20% test data
train_set= { 2:[] , 4:[] }
test_set=  { 2:[] , 4:[] }

train_data= full_data[:-int(len(full_data)*test_size)]  #80% data 
test_data= full_data[-int(len(full_data)*test_size):]  #last 20% data

for i in train_data:
    train_set[i[-1]].append(i[:-1]) #append the list without the '2's or '4's

for i in test_data:
    test_set[i[-1]].append(i[:-1])   #append the list without the '2's or '4's 

correct=0
total=0
benign=0
malig=0

for i in test_set:
    for j in test_set[i]:
        vote=k_nearest_neighbors(train_set,j,k=15) #predicting cancer is '2' or '4'
        if i==vote:
            if i==2:
                benign+=1
            else:
                malig+=1
            correct+=1
        total+=1


print('Accuracy of our algo is:',correct/total,benign,malig)



"""X=np.array(df.drop(['class'],1))   #class tells whether the tumor is benign or malignant
y=np.array(df['class'])

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

Kneigh= neighbors.KNeighborsClassifier()
Kneigh.fit(X_train,y_train)

accuracy= Kneigh.score(X_test,y_test)

print(accuracy)
"""
