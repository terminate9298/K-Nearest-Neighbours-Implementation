#Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Implementing KNN

def euclidean(X , Y):
  return np.sqrt(((X-Y)**2).sum())

class KNN:
  def __init__(self , use_distance = 'euc' , K = 'auto'):
    if use_distance == 'euc':
      self.distance = euclidean
    self.K = K
  
  def fit(self , X , Y):
    self.X = X
    self.Y = Y
    if self.K == 'auto':
      self.K = 7
  
  def predict_one(self , X):
    if X.shape[0] != self.X.shape[1]:
      raise Exception('Wrong Sized Data. Size Mismatch')
    else:
      dis = []
      for i in range(self.X.shape[0]):
        dis.append([self.distance(self.X[i] , X) , self.Y[i]])
      dis = np.array(dis)
      dis = dis[dis[:,0].argsort(kind = 'mergesort')]
      vals , count = np.unique(dis[:self.K], return_counts=True)
      return vals[np.argmax(count)]
  
  def predict(self , X):
    if X.shape[1] != self.X.shape[1]:
      raise Exception('Wrong Sized Data. Size Mismatch')
    else:
      dis = []
      for i in X:
        dis.append(self.predict_one(i))
      dis = np.array(dis)
      return dis
  def predict_accuracy(self , X , Y):
    dis = self.predict(X)
    acc = sum(1 for x,y in zip(dis,Y) if x == y) / len(dis)
    return acc

# Creating Dataset
class_1 = np.concatenate((np.random.normal(loc = -1, scale = 1 , size = (200,2)),np.array([1]*200).reshape(200,1)) , axis = 1)
class_2 = np.concatenate((np.random.normal(loc =  1, scale = 1 , size = (200,2)),np.array([-1]*200).reshape(200,1)) , axis = 1)
train, test = pd.DataFrame(np.concatenate((class_1[:160],class_2[:160]),axis = 0), columns = ('C_1','C_2','C')) , pd.DataFrame(np.concatenate((class_1[160:],class_2[160:]),axis = 0), columns = ('C_1','C_2','C')) 

# Predicting KNN  for K = 3
knn = KNN(K = 3)
knn.fit(train[['C_1','C_2']].values , train[['C']].values)
out = knn.predict_accuracy(test[['C_1','C_2']].values , test[['C']].values)
print ('Accuracy For K = 3 is {}'.format(out))

# Predicting KNN  for K = 5
knn = KNN(K = 5)
knn.fit(train[['C_1','C_2']].values , train[['C']].values)
out = knn.predict_accuracy(test[['C_1','C_2']].values , test[['C']].values)
print ('Accuracy For K = 5 is {}'.format(out))

# Predicting KNN  for K = 7
knn = KNN(K = 7)
knn.fit(train[['C_1','C_2']].values , train[['C']].values)
out = knn.predict_accuracy(test[['C_1','C_2']].values , test[['C']].values)
print ('Accuracy For K = 7 is {}'.format(out))
