from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

data = pd.read_csv("DebrisFlow.txt",sep='\t',header=None)

X = np.array(data[0])
Y = np.array(data[1])
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

y_hat = np.mean(Y)

reg = LinearRegression().fit(X,Y)

def compare(c,d):
    
    y_sk = reg.coef_[0] * X + reg.intercept_
    for cc,dd in zip(c,d):
        y_model = cc*X+dd
        plt.figure(figsize=(10,6))
        plt.scatter(X,Y)
        #plt.plot(X,y_sk,c='red',label="sklearn fit")
        plt.plot(X,y_model,c='green')
      
    plt.show()
    
    sk_rss = 1 -  np.sum(np.power(Y-y_sk,2))/np.sum(np.power(Y-y_hat,2))
    our_model_rss = 1 - np.sum(np.power(Y-y_model,2))/np.sum(np.power(Y-y_hat,2))
    
    '''print(f"sk learn r2 = {sk_rss}")
    print(f"our model r2 = {our_model_rss}")
    
    if our_model_rss > sk_rss:
        print("our model is doing better")
    elif our_model_rss < sk_rss:
        print("sk learn model is doing better")
    else:
        print("both models are fine!")'''
    
    