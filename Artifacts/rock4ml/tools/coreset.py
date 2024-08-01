import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
import heapq
# L_k_1 --> a vector holding the accumulated loss in the first k-1 rounds
# l_k --> a vector holding the current loss 
# _lambda --> the trade-off parameter
def dynamic_loss(L_k_1, l_k, _lambda):
  L_k = []
  for idx in range(len(L_k_1)):
    dynamioc_loss_t = (1 - _lambda) * L_k_1[idx] + _lambda * l_k[idx]
    L_k.append(dynamioc_loss_t)
  return np.array(L_k)

# pi should be set as a large value, since err(t) = P(t) will be maximual when pi approaches infinity 
def error_vector(L_k, pi):
    denominator = 0

    ERR = []

    for loss in L_k:
      denominator += np.exp(-1 * pi * loss)
    
    for loss in L_k:
      numerator = np.exp(-1 * pi * loss)
      P_t = numerator / denominator
      ERR.append(P_t)
      
    return np.array(ERR)

def coreset_identification(train_data, accuml_losses, pi, seed):
  np.random.seed(seed)
  error_signal = error_vector(accuml_losses, pi) # temperature
  
  random_number = np.random.rand(len(error_signal))
  random_number /= random_number.sum()
  coreset_indices = random_number < error_signal

  idxV = train_data[coreset_indices].index
  idxT = train_data.index.difference(idxV)

  return idxT, idxV