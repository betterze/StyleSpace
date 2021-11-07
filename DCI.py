# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Disentanglement, Completeness and Informativeness.
Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import pandas as pd


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  return scores

def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    print(i)
    model = GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)



def compute_importance_gbt2(x_train, y_train):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
#  test_loss = []
  for i in range(num_factors):
    print(i)
    model = GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
#    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss) #, np.mean(test_loss)






class DCI(): #only work for w and s
    def __init__(self,latent_path,attribute_path,p_threshold=2):
        self.p_threshold=p_threshold
        
        
        self.input=np.load(latent_path)
        self.attributes=pd.read_csv(attribute_path).iloc[:,:2]
        self.preprocessing()
        
        
        print('init DCI')
        
    def preprocessing(self):
        self.attrib_indices=self.attributes.columns
        self.num_samples=self.attributes.shape[0]
        self.keep_threshold=int(self.num_samples*0.05) #remove dimension each side less than 5%
        
        select11=self.attributes>0
        select1=select11.sum(axis=0)>self.keep_threshold
        
        select22=self.attributes<0
        select2=select22.sum(axis=0)>self.keep_threshold
        
        select=np.logical_and(select1,select2)
        
        self.attrib_indices2=self.attrib_indices[select]
        
        
        self.attributes2=self.attributes.loc[:,self.attrib_indices2]
        print('num_attribute',len(self.attrib_indices2))
        print('select attribute',self.attrib_indices2)
        
    def evaluate(self):
        train_loss=[]
        test_loss=[]
        importance_matrix = np.zeros(shape=[self.input.shape[1], len(self.attrib_indices2)],
                               dtype=np.float64)
        print(importance_matrix.shape)
        models=[]
        for i in range(len(self.attrib_indices2)):
            attribute=self.attributes2[self.attrib_indices2[i]]
            
            select1=attribute>np.percentile(attribute,100-self.p_threshold)
            select2=attribute<np.percentile(attribute,self.p_threshold)
            select=np.logical_or(select1,select2)
            x=self.input[select,:]
            y=self.attributes2.values[select,:]
            y[y>0]=1
            y[y<0]=0
            
            
            p=np.arange(len(y))
            np.random.shuffle(p)
            tmp=int(1/2*len(y))
            
            x_train=x[p[:tmp]]
            y_train=y[p[:tmp]]
            
            x_test=x[p[tmp:]]
            y_test=y[p[tmp:]]
            
            
            
            model = GradientBoostingClassifier(verbose=1)
            model.fit(x_train, y_train[:, i])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(np.mean(model.predict(x_train) == y_train[:, i]))
            test_loss.append(np.mean(model.predict(x_test) == y_test[:, i]))
            models.append(model)
        print(importance_matrix.shape)
        return importance_matrix,train_loss,test_loss,models




def Test(dci,save_path):
    with open(save_path, "rb") as fp:   #Pickling
        importance_matrix,train_loss,test_loss=pickle.load( fp)
    
    assert importance_matrix.shape[0] ==  dci.input.shape[1]
    assert importance_matrix.shape[1] == len(dci.attrib_indices2)
    
    scores={}
    scores["informativeness_train"] = np.mean(train_loss)
    scores["informativeness_test"] = np.mean(test_loss)
    
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the DCI of given latent codes')
    parser.add_argument('-latent_path',type=str,help='path to latent codes') 
    parser.add_argument('-attribute_path',type=str,help='path to attribute') 
    parser.add_argument('-save_path',type=str,help='path to save file')
    
    parser.add_argument('-mode',default='train',type=str,choices=['train','test']) 
    
    args = parser.parse_args()
    
    dci=DCI(args.latent_path,args.attribute_path)
    
    if args.mode=='train':
        importance_matrix,train_loss,test_loss,models=dci.evaluate()
        with open(args.save_path, "wb") as fp:
            pickle.dump([importance_matrix,train_loss,test_loss], fp)
            
    else:
        scores=Test(dci,args.save_path)
        print(scores)
        
    































