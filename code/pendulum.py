#%%
import numpy as np
import pickle

#%%
with open('data/pendulum.pkl', 'rb') as f:
    data = pickle.load(f)

#%%
X = data[0]
y = data[1]
#%%
len(X)
#%%
len(y)

#%%

#%%
