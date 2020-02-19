# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:44:44 2020

@author: hp
"""

from __future__ import print_function, division
from builtins import range, input


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam



path = "C://Users/hp/Pictures/recommendations/"
df = pd.read_csv(path+"ratings.csv")

N = df.userId.max()+1
M = df.movieId.max()+1

df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

#latent or hidden dimensionaity(k)=10
K = 15
mu = df.rating.mean()
epochs = 30
reg = 0.


u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u)
m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m)
u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m)
x = Dot(axes=2)([u_embedding, m_embedding])
x = Add()([x, u_bias, m_bias])
x = Flatten()(x)


model = Model(inputs=[u, m], outputs=x)
model.compile(loss='mse', 
              optimizer = SGD(lr=0.01, momentum=0.9),
              metrics=['mse'],
              )


r = model.fit(
        x = [df_train.userId.values, df_train.movieId.values],
        y = df_train.rating.values-mu,
        epochs=epochs,
        batch_size=160,
        validation_data = (
                [df_test.userId.values, df_test.movieId.values],
                df_test.rating.values-mu
                )
        )


plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()


