#%%
%matplotlib inline
%load_ext autoreload
%autoreload 2


#%%

import numpy as np
import pickle
import pbdlib as pbd
import matplotlib.pyplot as plt

#%%
with open('data/pendulum.pkl', 'rb') as f:
    data = pickle.load(f)

#%%
X = data[0]
y = data[1]
#%%
demos = []
for i in range(len(X)):
    tmp = []
    for j in range(len(X[i])):
        tmp += [list(X[i][j]) + list(y[i][j])]
    demos += [tmp]
#%%
demos = list(map(lambda x: np.array(x), demos))
#%%
model = pbd.HMM(nb_states=4, nb_dim=4)
#%%
model.init_hmm_kbins(demos)
#%%
model.em(demos, reg=1e-3)
#%%
model.sigma[0]
#%%
msg = model.predict(demos[0][0][0:3])

print(msg)
#%%
#gmm = pbd.GMM(nb_states=4, nb_dim=4, mu=model.mu, sigma=model.sigma, lmbda=model.lmbda, priors=model.priors)
#%%
#gmr = pbd.GMR(gmm, use_pybdlib_format=True)
#%%
#d = demos[0][:4]
#%%
#a = gmr.predict_GMM(d[0][0:3], [0, 1, 2], [3], predict=True)
#print(a)
#%%
