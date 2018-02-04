import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model , Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from catboost import CatBoostRegressor , CatBoostClassifier
import catboost
from sklearn.model_selection import train_test_split
import graphlab
from graphlab.toolkits.feature_engineering import CategoricalImputer
from keras.wrappers.scikit_learn import KerasClassifier as KC
from sklearn.model_selection import GridSearchCV

class FFKerasModel:
    
    def createModel(self):
        self.model = Sequential()
     
    def createLayer(self,output_dim , activation , input_dimen=0 , first = False):
        if (first):
            self.model.add(Dense(output_dim , input_dim = input_dimen , activation = activation))
        else:
            self.model.add(Dense(output_dim, activation = activation))
    
    def compileModel(self):
            self.model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])    

    def getModel(self):
        return self.model
        
def GridSearch(model,paramGrid,x,y):
    grid = GridSearchCV(estimator = model , param_grid = paramGrid , n_jobs = -1)
    gridResult = grid.fit(x,y)
    Best = gridResult.best_score_
    Params = gridResult.best_params_
    return Best , Params , gridResult
    

def GLImputer(Features,xtr):
    return graphlab.feature_engineering.create(xtr, CategoricalImputer(feature = Features))

def GLFitImputer(imputer , xtr):
    return imputer.transform(xtr)

def CatRegressor(itr,depth ,lr,loss,xtr,ytr,feat,xtest,ytest):
    model=catboost.CatBoostRegressor(iterations=itr, depth=depth, learning_rate=lr , loss_function=loss , use_best_model=True)
    model.fit(xtr, ytr ,cat_features=feat,eval_set=(xtest , ytest),plot=True)
    return model

def CatClassifier(itr,depth ,lr,loss,xtr,ytr,feat,xtest,ytest):
    model=catboost.CatBoostClassifier(iterations=itr, depth=depth, learning_rate=lr , loss_function=loss , use_best_model=True)
    model.fit(xtr, ytr ,cat_features=feat,eval_set=(xtest , ytest),plot=True)
    return model


def Spilt(X,y,split,rand):
    return train_test_split(X, y, train_size=split,random_state=rand)

def mergeGraphLab(dataset,dataset1,ON=None):
    return dataset.join(dataset1,on=ON)

def merge(dataset,dataset1,ON=None):
    return dataset.merge(dataset1,on=ON)
    
def checkNullValues(dataset):
    return dataset.isnull().sum()

def fillNA(dataset,value,Inplace = False):
    dataset.fillna(value,inplace = Inplace)
           
# CSV file to numpy array
class temp:
    
    def csv_to_numpy_array(filePath , delimiter):
        return np.genfromtxt(filePath, delimiter=delimiter, dtype='float32')

    def readData(filePath):
        return pd.read_csv(filePath) 

    def TokenizeData(Text,padding,truncating,MAX_WORDS):
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(Text)
        word2index = tokenizer.word_index
        num_words = len(word2index)
        print("Found unique tokens : " + num_words)    
        sequences = tokenizer.texts_to_sequences(Text)
        data = sequence.pad_sequences(sequences, maxlen=MAX_WORDS, padding=padding, truncating=truncating)
        return data    